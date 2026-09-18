# Flying on an Orange LTE router

Wi-Fi to a drone fails at a wall and you know why. A mobile link does not fail,
it *degrades*: RSRP slides as you fly away from the mast, SINR collapses when
the cell fills up at five o'clock, and the round trip goes from 40 ms to 600 ms
while the commands still arrive — late. A fly brain deciding at 20 Hz on the
other end of that is not flying the drone any more.

So FlyDrones reads the modem directly, scores the link, and lets the safety
governor act on it before it is gone.

```bash
flydrones link --link mock                       # try the whole thing with no router
flydrones link --link orange-airbox              # a real one, on its own address
flydrones link --link orange-flybox --link-target 192.168.8.100:8889 --csv link.csv
flydrones fly --drone tello --link orange-airbox --send
```

- [The boxes](#the-boxes)
- [What the prober reads](#what-the-prober-reads)
- [The numbers, and what they mean](#the-numbers-and-what-they-mean)
- [What the governor does about it](#what-the-governor-does-about-it)
- [Flying on it](#flying-on-it)
- [The honest part](#the-honest-part)

## The boxes

Orange ships two families of LTE box, and both answer on the LAN with nothing
installed:

| `--link` | address | what it is | API |
|---|---|---|---|
| `orange-airbox` | `192.168.8.1` | Orange Airbox 4G (Huawei E5783 / E5576 pocket router) | HiLink XML |
| `orange-flybox` | `192.168.8.1` | Orange Flybox 4G (Huawei B310s-22 / B525s-23) | HiLink XML |
| `orange-home-4g` | `192.168.8.1` | Orange Home 4G+ (Huawei B818 / B535) | HiLink XML |
| `orange-flybox-zte` | `192.168.0.1` | Orange Flybox (ZTE MF283 / MF286) | `goform` JSON |
| `huawei` / `zte` | as above | any other box of that family | |
| `auto` | | try Huawei, then ZTE | |
| `mock` | — | a scripted drive out of coverage | |

`flydrones link --list` prints the same table. If your box is on a different
address — Orange hands some out on `192.168.1.1` — pass `--link-host`.

**`mock` is not a toy.** It drives the whole chain — prober, scoring, the rules
that hold the drone and then land it — from full coverage down to nothing over
sixty seconds, so the behaviour can be seen, demonstrated and tested without a
router, the same way MiniFly stands in for the connectome.

## What the prober reads

Two halves, because they fail separately.

**The modem**, over the router's own LAN API. Huawei HiLink:

```
GET /api/webserver/SesTokInfo          a session cookie, if the firmware wants one
GET /api/device/signal                 rsrp, rsrq, sinr, rssi, band, cell_id, pci
GET /api/monitoring/status             ConnectionStatus, CurrentNetworkTypeEx
GET /api/net/current-plmn              operator and PLMN ("Orange F", 20801)
GET /api/monitoring/traffic-statistics throughput and uptime
```

ZTE, all in one:

```
GET /goform/goform_get_cmd_process?isTest=false&multi_data=1&cmd=lte_rsrp,lte_rsrq,lte_snr,...
```

Everything is a GET. Nothing in FlyDrones ever asks a router to change
anything: the worst this code can do to your connection is fetch a status page
once a second.

**The packets**, with `--link-target host:port` — a TCP handshake to the far
end, timed. Not ICMP: `ping` needs a raw socket (root on Linux, blocked in most
containers) and what matters is whether the *drone's own endpoint* is reachable
through the operator's network, not whether the router answers pings.

## The numbers, and what they mean

| metric | what it is | good | usable | edge |
|---|---|---|---|---|
| **RSRP** | how much of the cell's reference signal arrives: coverage | ≥ −85 dBm | −100 dBm | ≤ −110 dBm |
| **RSRQ** | how much of what arrives is the cell you want: interference and load | ≥ −10 dB | −15 dB | ≤ −18 dB |
| **SINR** | signal against noise: what the throughput actually follows | ≥ 20 dB | 5 dB | ≤ 0 dB |
| **RTT** | round trip to the far end | ≤ 60 ms | 250 ms | ≥ 500 ms |

Each is interpolated into 0..1 through `flydrones.link.probe.THRESHOLDS` —
every threshold on this page is in that table rather than buried in a
comparison — weighted (RSRP 0.4, SINR 0.4, RSRQ 0.2), then multiplied by the
latency and loss factors. The result is one **score from 0 to 1** and a grade:

```
0.75  excellent     0.55  good     0.35  fair     0.15  poor     below that, unusable
```

with the reasons that dragged it down, which is what the CLI prints and what
the governor puts in its log:

```
  12.0  0.31 poor      Orange F  LTE  B20  RSRP -107 dBm  RSRQ -16 dB  SINR 3 dB  cell 21102341  RTT 180 ms  loss 0%  (RSRP -107 dBm, SINR 3 dB)
```

## What the governor does about it

The link is part of the airframe when the airframe is on LTE, so it sits beside
the battery and the geofence in [`safety.py`](../src/flydrones/safety.py):

```yaml
safety:
  link:
    enabled: true
    hold_below: 0.35     # the drone stops going anywhere and hovers
    land_below: 0.15     # it lands, before the link decides for it
    timeout_s: 5.0       # no fresh reading for this long counts as no link
    grace_s: 2.0         # how long it has to stay bad: one bad poll is not a lost link
```

A poll that comes back bad once does nothing — mobile signal is noisy and a
single reading is not a lost link. A link that *stays* below `hold_below` for
`grace_s` makes the drone hover where it is; below `land_below`, or silent for
`timeout_s`, it lands. Both are logged, with the score and the reason:

```
lte poor (0.28) -> hold
lte unusable (0.04) -> land
```

The brain never sees any of this, and cannot override it — the governor is
downstream of the read-out, as it is for everything else.

## Flying on it

```bash
flydrones fly --drone tello --link orange-airbox --link-target 192.168.8.100:8889 --send
```

The probe runs on its own thread (an HTTP GET to a router takes tens to
hundreds of milliseconds and the control loop is 20 Hz, so it must not block
it), takes one reading before take-off, and the flight loop reads the last
verdict. `flydrones fly --link ...` refuses to take off if the router cannot be
reached at all: you asked to fly on LTE and there is no LTE.

A useful pattern before flying anywhere new: walk or drive the route with

```bash
flydrones link --link orange-flybox --link-target <drone endpoint> --csv survey.csv
```

and look at the worst score on the way. The summary line says how many readings
would have made the governor hold.

## The honest part

- **Not flight-tested by us.** Like the Tello, Crazyflie, MAVLink and ESP32
  adapters, this is written against the documented APIs and exercised against
  recorded responses and the mock. Please open an issue with a `--csv` when you
  try it on a real box.
- **Read-only, and unauthenticated where the firmware allows it.** Some HiLink
  firmwares want a login before `/api/device/signal`; if yours does, the probe
  reports the HTTP error rather than guessing at a password.
- **A good score is not permission to fly.** Flying beyond visual line of sight,
  and flying a drone commanded over a public mobile network, are regulated
  differently almost everywhere. The link score tells you what the radio is
  doing; it says nothing about what you are allowed to do, and nothing about
  the RC transmitter you should still be holding.
- **The uplink is the part that bites.** LTE is asymmetric and the drone's
  commands go up: a cell that shows a fine RSRP can still put 400 ms of jitter
  on a 20 Hz command stream at the wrong time of day. That is why the score
  includes the measured round trip and not only the modem's own numbers.
- **The mock is a smooth line, and real coverage is not.** It fades linearly to
  the edge of the cell with a little wobble; a real drive shows cliffs, handovers
  between cells and bands, and a sudden recovery when the box picks up B20.

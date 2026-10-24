"""KataGo sample schedules and sqrt-interpolated export-cycle LR scales.

EtaZero publishes every completed round. cycle_rounds groups those publication
intervals into one cycle without changing replay ratio or per-round work.
"""
import bisect
import math
import re


def points(text):
    entries = re.findall(r'\(\s*([0-9.]+[KMBkmb]?)\s*,\s*([+\-0-9.eE]+)\s*\)', text)
    residue = re.sub(r'\([^()]*\)', '', text).replace(',', '').strip()
    if not entries or residue:
        raise ValueError('LR points must be (samples,scale) pairs')
    result = []
    for count, value in entries:
        multiplier = {'K': 1e3, 'M': 1e6, 'B': 1e9}.get(count[-1].upper(), 1)
        number = float(count[:-1] if multiplier != 1 else count) * multiplier
        scale = float(value)
        if not number.is_integer() or not math.isfinite(scale) or scale < 0:
            raise ValueError('Invalid LR point')
        result.append((int(number), scale))
    if result[0][0] != 0 or any(b[0] <= a[0] for a,b in zip(result,result[1:])):
        raise ValueError('LR sample thresholds must start at zero and strictly increase')
    return result


def validate(options):
    if options['mode'] not in ('constant', 'auto', 'auto2', 'piecewise', 'cycle'):
        raise ValueError('Unknown LR schedule mode')
    if options['mode'] == 'piecewise': points(options['points'])
    if any(not math.isfinite(options[k]) or options[k] < 0 for k in ('auto2_factor','cycle_low','cycle_high')):
        raise ValueError('LR scales must be finite and nonnegative')
    if (options['cycle_rounds'] < 1 or options['cycle_up_rounds'] < 0 or options['cycle_down_rounds'] < 0 or
            options['cycle_up_rounds']+options['cycle_down_rounds'] > options['cycle_rounds']):
        raise ValueError('Invalid LR cycle durations')


def effective_scale(base, options, samples, completed_rounds=0):
    mode = options.get('mode', 'constant')
    if mode == 'constant': return base
    if mode == 'auto':
        thresholds = [550e6,1400e6,2500e6,3400e6,4200e6,4900e6,5500e6,6400e6]
        return [8,6,4,3,2,1.5,1,0.7,0.5][bisect.bisect_right(thresholds, samples)]
    if mode == 'auto2':
        thresholds = [20e6,40e6,70e6,100e6,140e6,180e6,220e6,260e6,300e6,350e6,400e6,450e6,500e6,550e6,600e6]
        return options['auto2_factor'] * [12,9,6,4,3,2,1.5,1,0.7,0.5,0.35,0.25,0.18,0.12,0.08,0.05][bisect.bisect_right(thresholds, samples)]
    if mode == 'piecewise':
        schedule = points(options['points'])
        return schedule[bisect.bisect_right([p[0] for p in schedule], samples)-1][1]
    period, up, down = (options[k] for k in ('cycle_rounds','cycle_up_rounds','cycle_down_rounds'))
    i = completed_rounds % period
    x = (i+1)/up if i < up else 1-(i-(period-down)+1)/down if i >= period-down else 1
    return ((1-x)*math.sqrt(options['cycle_low']) + x*math.sqrt(options['cycle_high']))**2

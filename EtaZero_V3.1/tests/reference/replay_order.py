"""KataGo gap-delaying repeat order, retained only for source comparisons (MIT)."""


def repeat_order(previous, rng):
    count = len(previous)
    k = (count * 2 + 1) // 3
    reservoir, order = previous[:k], []
    while k < count:
        index = rng.randrange(len(reservoir))
        reservoir[index], reservoir[-1] = reservoir[-1], reservoir[index]
        order.append(reservoir.pop())
        reservoir.append(previous[k])
        k += 1
    rng.shuffle(reservoir)
    return order + reservoir

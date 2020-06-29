from image import find_adjacents

def test_find_adjacents():
    data = [
        (
            ((0, 10), (20, 30)),
            ((0, 10), (20, 23), (27, 29), (40, 50))
        ),
        (
            ((0, 10), (20, 30), (40, 50)),
            ((0, 10), (42, 48))
        )
        ]
    
    answer = [
        {0: set((0,)), 1: set((1, 2)), None: set((3,))},
        {0: set((0,)), 1: set(), 2: set((1,))}
    ]

    for d, a in zip(data, answer):
        assert find_adjacents(*d) == a

test_find_adjacents()
from heapq import heappush, heappushpop

# heap sort by the first element of the tuple
# and then by the second element and etc
# so it's always a good idea to store data in a tuple

def kClosest(points: list[list[int]], k: int) -> list[list[int]]:

    hp = []
    ans = []

    for (px, py) in points:

        dist = -(px**2 + py**2)
        if len(hp) == k:
            heappushpop(hp, (dist, [px, py]))
        
        else:
            heappush(hp, (dist, [px, py]))
    
    for i in range(k):
        ans.append(hp[i][1])

    return ans


def findClosestElements(arr: list[int], k: int, x: int) -> list[int]:
    hp = []

    for num in arr:
        dist = -abs(num - x)
        if len(hp) == k:
            heappushpop(hp, (dist, -num))
        else:
            heappush(hp, (dist, -num))

    return sorted([-hp[i][1] for i in range(k)])

    
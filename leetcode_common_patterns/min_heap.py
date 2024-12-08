from heapq import heappush, heappushpop

# heap sort by the first element of the tuple

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


    
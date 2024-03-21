n = int(input())

arr = sorted(list(map(int, input().split())))

if n == 1:
    print(arr[0] * arr[0])
else:
    print(arr[0] * arr[-1])
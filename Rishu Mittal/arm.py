def arm(a):
    n = a
    r = 0
    while(n):
        n //= 10
        r = r+1
    
    n = a
    sum = 0
    while(n):
        x = n%10
        n //= 10
        sum += x**r
    return print("True") if(a == sum) else print("false")
    
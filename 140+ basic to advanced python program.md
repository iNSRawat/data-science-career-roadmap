# 140+ Basic to Advanced Python Programs - Interview Prep Kit

**Caption Summary:** 140+ Python Programs to Help You Ace Your Interviews. This power pack covers real interview patterns, strengthens logic, and is perfect for Data Analysts, Developers, and Freshers.

---

### Basic Programs
**Program 1: Write a Python program to print "Hello Python".**
```python
print("Hello Python")
```
*Output:* `Hello Python`

**Program 2: Write a Python program to do arithmetical operations addition and division.**
```python
# Addition
num1 = float(input("Enter the first number for addition: "))
num2 = float(input("Enter the second number for addition: "))
sum_result = num1 + num2
print(f"sum: {num1} + {num2} = {sum_result}")

# Division
num3 = float(input("Enter the dividend for division: "))
num4 = float(input("Enter the divisor for division: "))
if num4 == 0:
    print("Error: Division by zero is not allowed.")
else:
    div_result = num3 / num4
    print(f"Division: {num3} / {num4} = {div_result}")
```

**Program 3: Write a Python program to find the area of a triangle.**
```python
base = float(input("Enter the length of the base of the triangle: "))
height = float(input("Enter the height of the triangle: "))
area = 0.5 * base * height
print(f"The area of the triangle is: {area}")
```

---

### Variables & Conversions
**Program 4: Write a Python program to swap two variables.**
```python
a = input("Enter the value of the first variable (a): ")
b = input("Enter the value of the second variable (b): ")
print(f"Original values: a = {a}, b = {b}")
temp = a
a = b
b = temp
print(f"Swapped values: a = {a}, b = {b}")
```

**Program 5: Write a Python program to generate a random number.**
```python
import random
print(f"Random number: {random.randint(1, 100)}")
```

**Program 6: Write a Python program to convert kilometers to miles.**
```python
kilometers = float(input("Enter distance in kilometers: "))
conversion_factor = 0.621371
miles = kilometers * conversion_factor
print(f"{kilometers} kilometers is equal to {miles} miles")
```

---

### Calendar & Math
**Program 7: Write a Python program to convert Celsius to Fahrenheit.**
```python
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius} degrees Celsius is equal to {fahrenheit} degrees Fahrenheit")
```

**Program 8: Write a Python program to display calendar.**
```python
import calendar
year = int(input("Enter year: "))
month = int(input("Enter month: "))
cal = calendar.month(year, month)
print(cal)
```

**Program 9: Write a Python program to solve quadratic equation.**
*Standard Form:* `ax^2 + bx + c = 0`
*Formula:* `(-b ± sqrt(b^2 - 4ac)) / (2a)`
*(Code continues below)*

---

### Advanced Math & Logic
**Program 9 (Continued):**
```python
import math
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))
discriminant = b**2 - 4*a*c
if discriminant > 0:
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    print(f"Root 1: {root1}, Root 2: {root2}")
elif discriminant == 0:
    root = -b / (2*a)
    print(f"Root: {root}")
else:
    real_part = -b / (2*a)
    imaginary_part = math.sqrt(abs(discriminant)) / (2*a)
    print(f"Root 1: {real_part} + {imaginary_part}i")
    print(f"Root 2: {real_part} - {imaginary_part}i")
```

**Program 10: Write a Python program to swap two variables without temp variable.**
```python
a, b = 5, 10
a, b = b, a
print(f"After swapping: a = {a}, b = {b}")
```

---

### Conditionals
**Program 11: Check if a Number is Positive, Negative or Zero.**
```python
num = float(input("Enter a number: "))
if num > 0: print("Positive number")
elif num == 0: print("Zero")
else: print("Negative number")
```

**Program 12: Check if a Number is Odd or Even.**
```python
num = int(input("Enter a number: "))
if num % 2 == 0: print("Even number")
else: print("Odd number")
```

**Program 13: Check Leap Year.**
```python
year = int(input("Enter a year: "))
if (year % 400 == 0) and (year % 100 == 0): print(f"{year} is a leap year")
elif (year % 4 == 0) and (year % 100 != 0): print(f"{year} is a leap year")
else: print(f"{year} is not a leap year")
```

---

### Prime Numbers
**Program 14: Check Prime Number.**
```python
num = int(input("Enter a number: "))
flag = False
if num > 1:
    for i in range(2, num):
        if (num % i) == 0:
            flag = True
            break
    if flag: print(f"{num} is not a prime number")
    else: print(f"{num} is a prime number")
else: print(f"{num} is not a prime number")
```

---

### Factorials & Intervals
**Program 15: Display all prime numbers within an interval.**
```python
lower, upper = 1, 10
for num in range(lower, upper + 1):
    if num > 1:
        for i in range(2, num):
            if (num % i) == 0: break
        else: print(num)
```

**Program 16: Find the Factorial of a Number.**
```python
num = int(input("Enter a number: "))
factorial = 1
for i in range(1, num + 1):
    factorial *= i
print(f"Factorial of {num} is {factorial}")
```

---

### Sequences
**Program 17: Display the multiplication Table.**
```python
num = int(input("Table of: "))
for i in range(1, 11):
    print(f"{num} X {i} = {num*i}")
```

**Program 18: Print the Fibonacci sequence.**
```python
nterms = int(input("Terms? "))
n1, n2, count = 0, 1, 0
while count < nterms:
    print(n1)
    nth = n1 + n2
    n1, n2 = n2, nth
    count += 1
```

---

### Armstrong Numbers
**Program 19: Check Armstrong Number.**
```python
num = int(input("Enter a number: "))
order = len(str(num))
sum_pow, temp = 0, num
while temp > 0:
    digit = temp % 10
    sum_pow += digit ** order
    temp //= 10
if sum_pow == num: print(f"{num} is an Armstrong number")
else: print(f"{num} is not")
```

---

### Final Programs
**Program 20: Find Armstrong Number in an Interval.**
```python
lower, upper = 10, 1000
for num in range(lower, upper + 1):
    order = len(str(num))
    sum_val, temp = 0, num
    while temp > 0:
        digit = temp % 10
        sum_val += digit ** order
        temp //= 10
    if num == sum_val: print(num)
```

**Program 21: Find the Sum of Natural Numbers.**
```python
limit = int(input("Enter limit: "))
total = sum(range(1, limit + 1))
print(f"Sum: {total}")
```

---
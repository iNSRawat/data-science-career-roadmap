# 140+ Basic to Advanced Python Programs - Interview Prep Kit

**Caption Summary:** 140+ Python Programs to Help You Ace Your Interviews. This power pack covers real interview patterns, strengthens logic, and is perfect for Data Analysts, Developers, and Freshers.

---

### Basic Programs
**Program 1: Write a Python program to print "Hello Python".**
```python
print("Hello Python")
```
*Output:*
```text
Hello Python
```

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
*Output:*
```text
Enter the first number for addition: 10
Enter the second number for addition: 5
sum: 10.0 + 5.0 = 15.0
Enter the dividend for division: 10
Enter the divisor for division: 2
Division: 10.0 / 2.0 = 5.0
```

**Program 3: Write a Python program to find the area of a triangle.**
```python
base = float(input("Enter the length of the base of the triangle: "))
height = float(input("Enter the height of the triangle: "))
area = 0.5 * base * height
print(f"The area of the triangle is: {area}")
```
*Output:*
```text
Enter the length of the base of the triangle: 5
Enter the height of the triangle: 10
The area of the triangle is: 25.0
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
*Output:*
```text
Enter the value of the first variable (a): 5
Enter the value of the second variable (b): 9
Original values: a = 5, b = 9
Swapped values: a = 9, b = 5
```

**Program 5: Write a Python program to generate a random number.**
```python
import random
print(f"Random number: {random.randint(1, 100)}")
```
*Output:*
```text
Random number: 89
```

**Program 6: Write a Python program to convert kilometers to miles.**
```python
kilometers = float(input("Enter distance in kilometers: "))
conversion_factor = 0.621371
miles = kilometers * conversion_factor
print(f"{kilometers} kilometers is equal to {miles} miles")
```
*Output:*
```text
Enter distance in kilometers: 100
100.0 kilometers is equal to 62.137100000000004 miles
```

---

### Calendar & Math
**Program 7: Write a Python program to convert Celsius to Fahrenheit.**
```python
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius} degrees Celsius is equal to {fahrenheit} degrees Fahrenheit")
```
*Output:*
```text
Enter temperature in Celsius: 37
37.0 degrees Celsius is equal to 98.6 degrees Fahrenheit
```

**Program 8: Write a Python program to display calendar.**
```python
import calendar
year = int(input("Enter year: "))
month = int(input("Enter month: "))
cal = calendar.month(year, month)
print(cal)
```
*Output:*
```text
Enter year: 2024
Enter month: 5
      May 2024
Mo Tu We Th Fr Sa Su
       1  2  3  4  5
 6  7  8  9 10 11 12
13 14 15 16 17 18 19
20 21 22 23 24 25 26
27 28 29 30 31
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
*Output:*
```text
Enter coefficient a: 1
Enter coefficient b: -3
Enter coefficient c: 2
Root 1: 2.0, Root 2: 1.0
```

**Program 10: Write a Python program to swap two variables without temp variable.**
```python
a, b = 5, 10
a, b = b, a
print(f"After swapping: a = {a}, b = {b}")
```
*Output:*
```text
After swapping: a = 10, b = 5
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
*Output:*
```text
Enter a number: 7
Positive number
```

**Program 12: Check if a Number is Odd or Even.**
```python
num = int(input("Enter a number: "))
if num % 2 == 0: print("Even number")
else: print("Odd number")
```
*Output:*
```text
Enter a number: 4
Even number
```

**Program 13: Check Leap Year.**
```python
year = int(input("Enter a year: "))
if (year % 400 == 0) and (year % 100 == 0): print(f"{year} is a leap year")
elif (year % 4 == 0) and (year % 100 != 0): print(f"{year} is a leap year")
else: print(f"{year} is not a leap year")
```
*Output:*
```text
Enter a year: 2024
2024 is a leap year
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
*Output:*
```text
Enter a number: 29
29 is a prime number
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
*Output:*
```text
2
3
5
7
```

**Program 16: Find the Factorial of a Number.**
```python
num = int(input("Enter a number: "))
factorial = 1
for i in range(1, num + 1):
    factorial *= i
print(f"Factorial of {num} is {factorial}")
```
*Output:*
```text
Enter a number: 5
Factorial of 5 is 120
```

---

### Sequences
**Program 17: Display the multiplication Table.**
```python
num = int(input("Table of: "))
for i in range(1, 11):
    print(f"{num} X {i} = {num*i}")
```
*Output:*
```text
Table of: 5
5 X 1 = 5
5 X 2 = 10
5 X 3 = 15
5 X 4 = 20
5 X 5 = 25
5 X 6 = 30
5 X 7 = 35
5 X 8 = 40
5 X 9 = 45
5 X 10 = 50
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
*Output:*
```text
Terms? 5
0
1
1
2
3
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
*Output:*
```text
Enter a number: 153
153 is an Armstrong number
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
*Output:*
```text
153
370
371
407
```

**Program 21: Find the Sum of Natural Numbers.**
```python
limit = int(input("Enter limit: "))
total = sum(range(1, limit + 1))
print(f"Sum: {total}")
```
*Output:*
```text
Enter limit: 10
Sum: 55
```

---
# 🐍 Complete Python Cheatsheet

A comprehensive guide to the Python programming language, covering everything from basic structure to Object-Oriented Programming and advanced comprehensions.

---

## 🏗️ 1. Basic Structure

| Topic | Syntax / Keywords | Example | Output / Note |
| :--- | :--- | :--- | :--- |
| **Output** | `print()` | `print("Hello")` | Displays text |
| **Input** | `input()` | `x = input()` | Accepts user input |
| **Type Casting** | `int()`, `float()`, `str()` | `int("10")` | Converts data types |
| **Comment** | `#` | `# comment` | Single-line comment |
| **Multi-line** | `""" """` | `"""text"""` | Multi-line string/comment |

---

## 📊 2. Data Types

| Type | Keyword | Example |
| :--- | :--- | :--- |
| **Integer** | `int` | `a = 10` |
| **Float** | `float` | `x = 5.5` |
| **String** | `str` | `s = "hi"` |
| **Boolean** | `bool` | `True` / `False` |
| **List** | `list` | `[1, 2, 3]` |
| **Tuple** | `tuple` | `(1, 2, 3)` |
| **Set** | `set` | `{1, 2, 3}` |
| **Dictionary** | `dict` | `{"a": 1}` |
| **None Type** | `None` | `x = None` |

---

## ⚡ 3. Operators

| Category | Operators |
| :--- | :--- |
| **Arithmetic** | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| **Comparison** | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| **Logical** | `and`, `or`, `not` |
| **Assignment** | `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=` |
| **Membership** | `in`, `not in` |
| **Identity** | `is`, `is not` |

---

## 🔍 4. Conditional Statements

| Statement | Syntax |
| :--- | :--- |
| **If** | `if condition:` |
| **Elif** | `elif condition:` |
| **Else** | `else:` |
| **Pass** | `pass` |

**Example:**
```python
if x > 0:
    print("Positive")
```

---

## 🔄 5. Loops

| Loop Type | Syntax |
| :--- | :--- |
| **For Loop** | `for i in range():` |
| **While Loop** | `while condition:` |
| **Break** | `break` |
| **Continue** | `continue` |
| **Pass** | `pass` |

---

## 🔤 6. String Methods

| Method | Use |
| :--- | :--- |
| `len()` | Get Length |
| `lower()` | Convert to Lowercase |
| `upper()` | Convert to Uppercase |
| `strip()` | Remove leading/trailing spaces |
| `replace()` | Replace specific text |
| `split()` | Convert string to list |
| `" ".join()` | Join list elements into string |
| `find()` | Search for substring |
| `count()` | Count specific occurrences |
| `startswith()` | Check if starts with value |
| `endswith()` | Check if ends with value |

**Slicing:**
- `s[start:end:step]`
- `s[::-1]` (Reverse string)

---

## 📜 7. List Methods

| Method | Purpose |
| :--- | :--- |
| `append()` | Add element to end |
| `extend()` | Append multiple elements |
| `insert()` | Insert at specific index |
| `remove()` | Remove specific element |
| `pop()` | Remove by index |
| `clear()` | Empty the entire list |
| `index()` | Find index of a value |
| `count()` | Count value occurrences |
| `sort()` | Sort list in-place |
| `reverse()` | Reverse list in-place |

**List Comprehension:**
- `[x for x in iterable]`
- `[x for x in iterable if condition]`

---

## 🔒 8. Tuple

| Feature | Description |
| :--- | :--- |
| **Immutable** | Content cannot be modified after creation |
| **Methods** | `count()`, `index()` |

---

## 🎯 9. Set Methods

| Method | Purpose |
| :--- | :--- |
| `add()` | Add single element |
| `update()` | Add multiple elements |
| `remove()` | Remove element (errors if missing) |
| `discard()` | Remove element (safe if missing) |
| `union()` | Combine sets (`|`) |
| `intersection()`| Common elements (`&`) |
| `difference()` | Difference between sets (`-`) |
| `symmetric_difference()` | Unique elements in both (`^`) |

---

## 📖 10. Dictionary Methods

| Method | Purpose |
| :--- | :--- |
| `keys()` | Get all keys |
| `values()` | Get all values |
| `items()` | Get key-value pairs |
| `get()` | Safe access to a key |
| `update()` | Update or add items |
| `pop()` | Remove specific key |
| `clear()` | Empty the dictionary |

**Dictionary Comprehension:**
- `{k:v for k,v in iterable}`

---

## ⚙️ 11. Functions

| Concept | Syntax |
| :--- | :--- |
| **Define Function** | `def name():` |
| **Return** | `return value` |
| **Default Argument** | `def f(x=10):` |
| ***args** | Variable positional arguments |
| ****kwargs** | Variable keyword arguments |
| **Lambda** | `lambda x: x*x` |

---

## 🔁 12. Recursion

| Concept | Syntax |
| :--- | :--- |
| **Recursive Function**| `def f(n):` |
| **Base Case** | `if n == 0:` |

---

## 📦 13. Modules

| Keyword | Use |
| :--- | :--- |
| `import module` | Import full module |
| `from mod import X`| Import specific name |
| `as` | Assign an alias |

**Common Modules:**
- `math`, `random`, `os`, `sys`, `string`, `datetime`

---

## 📂 14. File Handling

| Mode | Meaning |
| :--- | :--- |
| **r** | Read (Default) |
| **w** | Overwrite/Write |
| **a** | Append |

**Methods:**
- `read()`, `readline()`, `write()`, `close()`

**Safe Context Manager:**
```python
with open("file.txt") as f:
    content = f.read()
```

---

## ⚠️ 15. Exception Handling

| Block | Syntax |
| :--- | :--- |
| **Try** | `try:` (Code to test) |
| **Except** | `except:` (Handle error) |
| **Finally** | `finally:` (Always runs) |
| **Raise** | `raise` (Trigger exception) |

**Common Exceptions:**
- `ZeroDivisionError`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `FileNotFoundError`

---

## 🏛️ 16. OOP (Object Oriented Programming)

| Concept | Syntax |
| :--- | :--- |
| **Class** | `class Name:` |
| **Constructor** | `__init__(self)` |
| **Self** | `self` (Refers to instance) |
| **Object** | `obj = Class()` |

**Inheritance:**
```python
class Child(Parent):
    pass
```

---

## 🛠️ 17. Built-in Functions

| Function | Purpose |
| :--- | :--- |
| `len()` | Length of object |
| `max() / min()` | Maximum / Minimum |
| `sum()` | Sum of elements |
| `sorted()` | Returns sorted list |
| `map()` | Apply function to iterable |
| `filter()` | Filter elements |
| `zip()` | Combine iterables |
| `enumerate()` | Provides index + value |
| `eval()` | Evaluates a string as code |
| `abs()` | Absolute value |
| `round()` | Round number |

---

## 🧱 18. Comprehensions

| Type | Syntax |
| :--- | :--- |
| **List** | `[x for x in range()]` |
| **Set** | `{x for x in range()}` |
| **Dictionary**| `{k:v for k,v in iterable}` |

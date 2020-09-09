# CC6204 Autocorrect

Basic tooling for autocorrecting programming homeworks for the CC6204: Deep Learning course at the University of Chile

## Instalation

`pip install cc6204 ...`

## Usage
### Instantiate a corrector object:
```python
from cc6204 import AutoCorrect

corrector = AutoCorrect(host=<host>, port=<value>)
```

### Submit an answer for revision:
```python
corrector.sumbit(task=1, test=1, token=<token>, answer=...)
```
Submit will raise a `FailedTest` exception when the submited answer is not what we were expecting. If you want to know which values you submited were labeled incorrect, catch the exception and get the `msg` attribute as follows:
```python
from cc6204 import FailedTest

revision = None
try:
    corrector.sumbit(task=1, test=1, token=<token>, answer=...)
except FailedTest as e:
    revision = e.msg

print(revision)
```

If you get all correct values, then no exception is raised and a `"Correct Test!"` string is printed to console.

### Get the tests being used for revision:
```python
test_input = corrector.get_test_data(task=1, test=1, token=<token>)
```
Use this `test_input` values as an input to you implementation and send the output as an answer to the `submit` method.


### How to use it in your homework

First install the library and instantiate a corrector object with the `host`, `port` and `token` values given in U-cursos.

```python
from cc6204 import AutoCorrect, FailedTest

corrector = AutoCorrect(host=<host>, port=<value>)
```

Let's say you are working on Question 2 in Homework 1. You would first implement the given question. Then ask the library for the test set to evaluate your implementation using the `get_test_data` method.
```python
def my_implementation(some_input):
    # my hard work
    ...
    return output

test_input = corrector.get_test_data(task=1, test=2, token=<token>)
```

Use the returned value as an input to your implementation and save the output value. Pass the output value to call the `submit` method and review your mistakes.
```python
my_answer = my_implementation(test_input)
revision = None
try:
    corrector.sumbit(task=1, test=2, token=<token>, answer=my_answer)
except FailedTest as e:
    # this is a mask with the values you missed
    revision = e.msg

if revision is not None:
    # you made mistakes
    print(revision)
# if revision is None, then a "Correct Test!" printed in console and your implementation passed the test
```

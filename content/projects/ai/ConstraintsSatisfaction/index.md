+++
title = "Constraint Satisfaction Problem"
summary = "Solving logic puzzle in Symbolic AI using the python-constraint API."
description = ""
featuredImage = "featured.jpg"
tags = ["Logic Puzzle", "Symbolic AI"]
categories = ["AI"]
collections = [""]
weight = 11
draft = false
+++

## Symbolic AI
Symbolic Artificial Intelligence (AI) is a subfield of AI that focuses on the processing and manipulation of symbols or concepts, rather than numerical data. 

The goal of Symbolic AI is to build intelligent systems that can reason and think like humans by representing and manipulating knowledge and reasoning based on logical
rules.

## Constraint Satisfaction
Constraint satisfaction is one of the areas in which Symbolic AI has been successful. It is the process of solving a problem by satisfying certain constraints or conditions. For example, you are asked to color a map using red, yellow, and green only, and you can’t use the same color for two adjacent areas.

## Solving Logic Puzzle “What are the Contracts of the Car Renting Customers?”

There are 5 car renting contracts. Each has duration, a pickup location, a car brand and a customer associated to it. The goal is to find out for all contracts, what the customer, duration, car brand and pickup location is (again given the fact that each attribute is unique amongst the contracts).

### Attributes

- The pickup locations are: Brownfield, Durham, Iowa Falls, Los Altos and Redding.
- The car brands are: Dodge, Fiat, Hyundai, Jeep and Nissan.
- The contract duration are 2, 3, 4, 5 and 6 days.
- The customer names are Freda, Opal, Penny, Sarah and Vicky.

### Hints

1. The contracts of Vicky, the one with pickup location in Los Altos, the one with pickup location in Durham and the one with the Fiat are all different contracts.
2. The contract with the Jeep is not in Iowa Falls.
3. The contract of Vicky and the one with the Nissan are picked up in either Los Altos or Redding.
4. Penny’s contract is not for 6 days.
5. The contract in Iowa Falls is for 5 days.
6. The contract with the Durham is 3 days longer than the contract of Opal.
7. Of the contract with Nissan and the 2 day contract, one is picked up in Redding and the other is Freda’s contract.
8. The contract with the Jeep is not for 6 days.
9. The contract of Opal is 1 day longer than the one with the Hyundai.

{{< button href="https://colab.research.google.com/drive/1JLuE3ckVHpew8_ghgfiGMi6vBNKPjip6" target="_blank" color="color-colab" >}}
{{< icon "colab" >}} View on Google Colab
{{< /button >}}

To solve this Constraint Satisfaction Problem (CSP), I used the `python-constraint` API. After reading the problem description, the first thing is create variables for each attribute (`customers`, `locations`, `brands`, and `durations`) and add possible values for each attribute. Each contract has a unique combination of these four attributes.


```py
locations = ["Brownfield", "Durham", "Iowa Falls", "Los Altos", "Redding"]
brands = ["Dodge", "Fiat", "Hyundai", "Jeep", "Nissan"]
durations = ["2 day", "3 day", "4 day", "5 day", "6 day"]
customers = ["Freda", "Opal", "Penny", "Sarah", "Vicky"]
```


Then I initialize the `Problem` and created a list of all variables. Use the `addVariables` method to assign each variable a possible contract number between 2 and 6 (also representing the five durations). Apply the `AllDifferentConstraint` to ensure each attribute is unique amongst the contracts.


```py
minn, maxn = 2, 6
problem = Problem()

# Value of a variable is the number of a contract with corresponding property
variables = locations + brands + durations + customers
problem.addVariables(variables, range(minn, maxn+1))

# Each attribute is unique amongst the contracts
for vars_ in (locations, brands, durations, customers):
    problem.addConstraint(AllDifferentConstraint(), vars_)
```


The next thing is translate every hint into a constraint using lambda functions and then got only one solution:


```py
# Add constraint for durations
problem.addConstraint(lambda d2, d3, d4, d5, d6: d2 == 2 and d3 == 3 and d4 == 4 and d5 == 5 and d6 == 6, ("2 day", "3 day", "4 day", "5 day", "6 day"))

# 1. The contracts of Vicky, the one with pickup location in Los Altos,
# the one with pickup location in Durham and the one with the Fiat are all different contracts.
problem.addConstraint(lambda v, la, d, f: v != la != d != f, ("Vicky", "Los Altos", "Durham", "Fiat"))

# 2. The contract with the Jeep is not in Iowa Falls.
problem.addConstraint(lambda j, i: j != i, ("Jeep", "Iowa Falls"))

# 3. The contract of Vicky and the one with the Nissan are picked up in either Los Altos or Redding.
problem.addConstraint(lambda v, n, la, r: (v == la and n == r) or (v == r and n ==la), ("Vicky", "Nissan", "Los Altos", "Redding"))

# 4. Penny’s contract is not for 6 days.
problem.addConstraint(lambda p, d: p != d, ("Penny", "6 day"))

# 5. The contract in Iowa Falls is for 5 days.
problem.addConstraint(lambda i, d: i == d, ("Iowa Falls", "5 day"))

# 6. The contract with Durham is 3 days longer than the contract of Opal.
problem.addConstraint(lambda d, o: d == o + 3, ("Durham", "Opal"))

# 7. Of the contract with Nissan and the 2-day contract, one is picked up in Redding and the other is Freda’s contract.
problem.addConstraint(lambda n, d, r, f: (n == r and d == f) or (n == f and d == r), ("Nissan", "2 day", "Redding", "Freda"))

# 8. The contract with the Jeep is not for 6 days.
problem.addConstraint(lambda j, d: j != d, ("Jeep", "6 day"))

# 9. The contract of Opal is 1 day longer than the one with the Hyundai.
problem.addConstraint(lambda o, h: o == h + 1, ("Opal", "Hyundai"))
```

```py
solutions = problem.getSolutions()

for solution in solutions:
    print(solution)
```


Here I added another part to format the solution in a more readable way: 


```py
# Print the solution in a more readable way
for solution in solutions:
    print("Solution:")
    for c in range(minn, maxn+1):
        if c in solution.values():
            print([k for k, v in solution.items() if v == c])
```

```
Solution:
['2 day', 'Redding', 'Vicky', 'Hyundai']
['Jeep', 'Opal', '3 day', 'Brownfield']
['Los Altos', 'Nissan', 'Freda', '4 day']
['5 day', 'Iowa Falls', 'Fiat', 'Penny']
['6 day', 'Durham', 'Dodge', 'Sarah']
```

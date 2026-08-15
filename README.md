# nl-calculator

A simple calculator, but in natural language.

## Example

```raw
[user]
"What is 45 times 2?"
```

```raw
[tool-call]
{
  "operator": "multiply",
  "left": 45,
  "right": 2,
}
```

```raw
[assistant]
"45 times 2 is 90."
```

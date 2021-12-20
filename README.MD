# Example
## mode==0
```
Input: 
  sentence: I have a lot of [MASK] [MASK]

Output: 
  i have a lot of questions.
  i have a lot of money.
  i have a lot of friends.
```
## mode==1

```
Input:
  sentence: have a lot of
  left mask number:1
  right mask number:2

Output:
  i have a lot of questions.
  i have a lot of money.
  we have a lot of questions.
```
import random
#min_value=int(input("Enter the minimun value:"))
#max_value=int(input("Enter the maximum value:"))
min_value=1
max_value=6
again=True

while again:
    (print(random.randint(min_value,max_value)))
    

    another_role=("want to role again")

    if another_role.lower()=="yes" or another_role.lower()=="y":
        continue
    else:
        break

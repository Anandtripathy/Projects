import random
# Instructions:
print("Winning Rules of the Rock paper scissor game as follows: \n"
                                +"Rock vs paper->paper wins \n"
                                + "Rock vs scissor->Rock wins \n"
                                +"paper vs scissor->scissor wins \n")

while True:
    print("Enter Choice\n 1. Rock \n 2. Paper \n 3. scissor \n")

    choice=int(input("Enter from choice: "))
    
    while choice > 3 or choice < 1:
        choice=int(input("Enter the valid choice: "))
        
    if choice==1:
        choice_name="Rock"
    elif choice==2:
        choice_name="Paper"
    else:
         choice_name="scissor"

    #print your choice
    print("User choice is: "+choice_name)

#Computer Turn 
    print("Now computer turn...")

    comp_choice=random.randint(1,3)

    while comp_choice==choice:
        comp_choice=random.randint(1,3)
    if comp_choice==1:
        comp_choice_name="Rock"
    elif comp_choice==2:
        comp_choice_name="Paper"
    else:
        comp_choice_name="scissor"

    print("Computer choice is: "+comp_choice_name)
    print(choice_name +  " V/s " + comp_choice_name)

#condition for winning:
    if((choice==1 and comp_choice==2) or (choice==2 and comp_choice==1)):
        print("Paper wins =>", end="")
        result="paper"
    elif((choice==1 and comp_choice==3) or (choice==3 and comp_choice==1)):
        print("Rock wins=>",end="")
        result="Rock"
    else:
        print("scissor wins =>", end="")
        result="scissor"

#printing either user or computer wins
    if result ==choice_name:
        print("<==User wins ==>")
    else:
        print("<== Computer wins ==>")
        
    print("Do you want to play again:? (Y/N)")
    ans=input()

    if ans=='n' or ans=='N':
        print("Thank you")
    
    break






        
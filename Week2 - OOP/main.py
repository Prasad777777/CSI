# main.py

import random
from name_validator import get_valid_name

# Lambda functions
square = lambda x: x * x
square_root = lambda x: int(x ** 0.5)

# Ask if user wants to continue
def ask_continue():
    choice = input("\nDo you want to continue the quiz? (yes/no): ").strip().lower()
    return choice in ['yes', 'y']

# Ask which quiz user wants
def choose_quiz_type():
    while True:
        choice = input("\nWhich quiz do you want to play? Type 'square' or 'square root': ").strip().lower()
        if choice in ['square', 'square root']:
            return choice
        else:
            print("⚠️ Please enter a valid choice: 'square' or 'square root'.")

# Ask square question
def ask_square_question(score):
    num = random.randint(1, 100)
    try:
        answer = int(input(f"\nWhat is the square of {num}? "))
        if answer == square(num):
            score += 10
            print("✅ Correct! +10 points.")
        else:
            score -= 5
            print(f"❌ Wrong! The correct answer was {square(num)}. -5 points.")
    except ValueError:
        print("⚠️ Please enter a valid number.")
    return score

# Ask square root question
def ask_square_root_question(score):
    root = random.randint(1, 100)
    perfect_square = square(root)
    try:
        answer = int(input(f"\nWhat is the square root of {perfect_square}? "))
        if answer == root:
            score += 10
            print("✅ Correct! +10 points.")
        else:
            score -= 5
            print(f"❌ Wrong! The correct answer was {root}. -5 points.")
    except ValueError:
        print("⚠️ Please enter a valid number.")
    return score

# === Main Program ===
name = get_valid_name()
print(f"\nWelcome, {name}!")

score = 0
while True:
    quiz_type = choose_quiz_type()

    if quiz_type == 'square':
        score = ask_square_question(score)
    else:
        score = ask_square_root_question(score)

    print(f"Your current score: {score}")

    if not ask_continue():
        print(f"\nThanks for playing, {name}! Your final score is: {score}")
        break

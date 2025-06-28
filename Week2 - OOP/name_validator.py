# name_validator.py

# Custom exception class
class InvalidNameError(Exception):
    def __init__(self, message="Invalid name. Please enter only alphabetic characters (A-Z or a-z)."):
        super().__init__(message)

# Name validation function
def get_valid_name():
    while True:
        name = input("Enter your name: ").strip()
        try:
            if not name.isalpha():
                raise InvalidNameError()
            return name
        except InvalidNameError as e:
            print(f"❌ {e}")

from person import Person
class Student(Person): # Inheriting from Person class and extends the behaviour of Person
    def __init__(self, *args):  # CAN'T USE INIT cos it's in previous init Person class
        super().__init__(*args)
        
        self.subject = None  #  Ahhhhhh the *args means you can use init in this class as well
    
    def enroll(self, subject):
        self.subject = subject
        
    def get_full_name(self): # Redefine full name function, completely changed name of function
        if self.subject is None:
            return super().get_full_name()
        else:
            full_name = f"{self.name} {self.last_name} enrolled in {self.subject}"
            return full_name
        
        
if __name__ == "__main__":  # This stops you running all of code in Person and only imports useful bits
    print('Student.py is being run')
    someone = Student('Jane', 'Doe')
    #someone.enroll('math')
    someone.calculate_age(2000)
    print(someone.get_full_name())  # Every time we run student on import it runs all of them. Python runs entire file.
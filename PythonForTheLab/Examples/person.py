class Person:  # Not inheriting from anyone
    def __init__(self, name, last_name): # init runs when we initialise object, self will always be there in every function, referring to object itself and always run explicitly
        self.name = name  #  self. is critical. You're accessing variable to whole object.
        self.last_name = last_name  # init is the only one you can guarantee will run
        self.birth_year = None  # This is initialized when using 'Calculate Age'
        
    def get_full_name(self):
        full_name = f"{self.name} {self.last_name}" # f string. Putting f in front of string and then you can insert variables in.
        return full_name
    
    def calculate_age(self, birth_year): #we don't know age so need to add it
        self.birth_year = birth_year
        print(f"{self.name} is {2024-birth_year} years old")
        
    def over_age(self):
        if self.birth_year is None:
            return "I don't know"
        if 2024 - self.birth_year > 18:
            return True
        else:
            return False
        
        
if __name__ == "__main__":
    print('Person.py is being run')
    me = Person('Aquiles', 'Carattino')  # me is an object
    #print(me.name)  # can access information in object with this. These were defined in self.name and self.last_name
    #print(me.last_name)
    print(me.get_full_name())
    me.calculate_age(1986)
    #print(me.birth_year)
    print(me.over_age())


    you = Person('John', 'Smith')
    #you.name = 'John'
    #print(you.name)
    #print(you.last_name)
    #you.calculate_age(1995)

    print(you.over_age())
from zoneinfo import available_timezones
from scipy.optimize import minimize
import numpy

# operators = ["KC", "DH", "HB", "SC", "KS", "NK"]
week = ["M", "T", "W", "R", "F"]

operators_reverse_order = ["NK", "KS", "DH", "KC", "HB", "SC"]
wage_reverse_order = [11.30, 10.80, 10.10, 10.00, 9.90, 9.80]
availabilities = {"KC": [6,0,6,0,6], "DH": [0,6,0,6,0], "HB": [4,8,4,0,4], "SC": [5,5,5,0,5], "KS": [3,0,3,8,0] , "NK": [0,0,0,6,2] }


operators_wages = {"KC": 10.00, "DH": 10.10, "HB": 9.90, "SC": 9.80, "KS": 10.80, "NK": 11.30}
student_status = {"KC": "u", "DH": "u", "HB": "u", "SC": "u", "KS": "g", "NK": "g"}
operators_hours = [0, 0, 0, 0, 0, 0]
hours_remaining = 70


def optimize_by_operator(ops):
    
    cost = 0
    total_hours_worked = 0
    lists = []
    hours_per_day = [0,0,0,0,0]
    full_days = [False, False, False, False, False]
    for op in ops:
        availability = availabilities[op]
        
        hours_remaining = 0
        if student_status[op] == "u":
            hours_remaining = 8
        elif student_status[op] == "g":
            hours_remaining = 7
        
        working_hours = [0,0,0,0,0]
        
        for index in range(len(week)):
            if full_days[index] == False:
                if availability[index] < hours_remaining:
                    if 14 - availability[index] < hours_per_day[index]:
                        # i = 14 - hours_per_day[index]
                        # availability[index] -= i
                        hours = 14 - hours_per_day[index]
                        availability[index] -= hours
                        i = availability[index] * operators_wages[op]
                        cost += i
                        working_hours[index] += hours
                        total_hours_worked += hours
                        hours_per_day[index] += hours
                        availability[index] -= hours
                        hours_remaining -= hours
                        full_days[index] = True
                        
                    else:
                        i = availability[index] * operators_wages[op]
                        cost += i
                        working_hours[index] += availability[index]
                        total_hours_worked += availability[index]
                        hours_per_day[index] += availability[index]
                        hours_remaining -= availability[index]
                        availability[index] = 0
                else:
                    i = hours_remaining * operators_wages[op]
                    cost += i
                    working_hours[index] += hours_remaining
                    total_hours_worked += hours_remaining
                    hours_per_day[index] += hours_remaining
                    availability[index] -= hours_remaining
                    hours_remaining = 0
                    lists.append(working_hours)
                    break
                
    hours_left = 70 - total_hours_worked
    while hours_left > 0:
        print("")
        hours_left -= 1
    return cost, total_hours_worked, lists

print(optimize_by_operator(operators_reverse_order))




#Objective function: minimize cost while completing hours, meeting minimum requirements, time availabilities
def cost_function():
    cost = 0
    for o in operators_wages:
        if student_status[o] == "u":
            cost += (8 * operators_wages[o])
            print(cost)
        elif student_status[o] == "g":
            cost += (7 * operators_wages[o])
            print(cost)
        else:
            print("Failed")
    return cost
    
# minimize(cost_function(), 50)


# this function takes in the wages dictionary and returns the wages in reverse order
def order_by_wages_reversed(dictionary):
    values = []
    for value in dictionary.values():
        values.append(value)
    values.sort()
    values.reverse()
    return values

# this function takes in the wages dictionary and returns the wages in regular order
def order_by_wages(dictionary):
    values = []
    for value in dictionary.values():
        values.append(value)
    values.sort()
    return values

# by day, by person (wage), find holes

def optimize_by_day():
    for day in week:
        day_hours_remaining = 14
        wages = order_by_wages_reversed(operators_wages)
        
        
    return



import csv
import random

# This func gen Mac addresses that rep each state polling unit!

def generate_nigerian_macs():
    states = ['Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno', 'Cross River',
              'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Kano',
              'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun',
              'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara', 'FCT']

    with open('nigeria_macs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write states as headers  
        writer.writerow(states)

        # 176,974 - Polling units in Nigeria,
        # so change this when gen authentic data
        # for now leave 100 for testing
        for i in range(100):
            row = []
            for state in states:
                mac = ':'.join(['{:02x}'.format(random.randint(0, 255), 'x') for i in range(6)])
                row.append(mac)
            writer.writerow(row)


generate_nigerian_macs()
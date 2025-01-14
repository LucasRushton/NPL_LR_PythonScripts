import yaml

with open('Examples/config.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    
print(data)
print(data['Scan']['step'])
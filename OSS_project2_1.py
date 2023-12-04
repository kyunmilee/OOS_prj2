import pandas as pd

data = pd.read_csv("2019_kbo_for_kaggle_v2.csv")

def topten(years, fields):
    top_each_year = {}
    
    for year in years:
        year_data = data[data['year'] == year]
        year_fields = pd.DataFrame(index=range(1, 11))
        
        for field in fields:
            top_players = year_data.nlargest(10, field)[['batter_name']]
            top_list = top_players['batter_name'].tolist()
            year_fields[field] = pd.Series(top_list, index=range(1, len(top_list) + 1))
        
        top_each_year[year] = year_fields
    
    return top_each_year
        
        
def high_each_poisition(year):
    year_data = data[data['year'] == year]
    positions = year_data['cp'].unique()
    
    top_players = []
    
    for position in positions:
        position_player = year_data[year_data['cp'] == position]
        top_player = position_player.loc[position_player['war'].idxmax()]
        top_players.append({
            'batter_name': top_player['batter_name'],
            'cp': top_player['cp'],
            'war': top_player['war']
        })
    two_result = pd.DataFrame(top_players)
        
    return two_result.to_string(index=False)


def high_corr_salary(fields):
    relation_salary = fields + ['salary']
    
    correlation_salary = data[relation_salary].corr()['salary'].drop('salary')
    
    high_corr_rela = correlation_salary.idxmax()
    high_corr_val = correlation_salary.max()
    
    return correlation_salary, high_corr_rela, high_corr_val

fields = ['H', 'avg', 'HR', 'OBP']
years = [2015, 2016, 2017, 2018]
first_result = topten(years, fields)

for i, k in first_result.items():
    print("IN {}".format(i))
    print(k)
    print("\n\n")
    
print("*" * 36)
print("\n")

high_war_2018 = high_each_poisition(2018)
print(high_war_2018)
print("\n\n")

print("*" * 36)
print("\n\n")

field_three = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
correlation, highest_relation, highest_value = high_corr_salary(field_three)

print("연봉과 각 분야의 관계:")
print(correlation.to_string())
print("\n연봉과 가장 높은 상관관계: {} \n상관계수: {}".format(highest_relation, highest_value))


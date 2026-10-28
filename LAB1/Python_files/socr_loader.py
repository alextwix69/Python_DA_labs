import pandas as pd

tables = pd.read_html("baseball.html")

df = next(
    table for table in tables
    if {"Name", "Team", "Position", "Age"}.issubset(table.columns)
)

assert df.shape == (1034, 6), df.shape

df = df.drop(columns=['Name', 'Team'])
df['Position'] = df['Position'].replace({
    'Starting_Pitcher': 1.,
    'Catcher': 2., 
    'First_Baseman': 3., 
    'Outfielder': 4.,
    'Relief_Pitcher': 5., 
    'Second_Baseman': 6., 
    'Third_Baseman': 7.,
    'Shortstop': 8.,
    'Designated_Hitter': 9.
})

df.to_csv("baseball.csv", index=False)

print(df)
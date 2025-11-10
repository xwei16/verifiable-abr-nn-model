import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('log_sim_bb_cleaned_norway.csv')

# Get unique values from the 'QoE_2' column
unique_qoe_values = df['QoE_2'].unique()

# Print them
for val in unique_qoe_values:
    print(val)


# 2.772588722239781
# 0.0
# 1.83258146374831
# 3.638316886832339
# 2.7725887222397816
# 1.8325814637483104
# 4.50258359721299
# 1.8325814637483095
# 5.325175654050906


brs = [300,750,1200,1850,2850,4300]

def qoe(br1, br2):
    # Example function to calculate QoE based on two bitrates
    # This is a placeholder; the actual calculation may vary
    last_q = np.log(br1 / brs[0])
    cur_q = np.log(br2/ brs[0])
    qoe_2 =  last_q + cur_q - np.abs(last_q - cur_q)
    return qoe_2


qoe_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.83258146374831, 1.8325814637483098, 1.8325814637483102, 1.8325814637483102, 1.8325814637483102, 0.0, 1.8325814637483098, 2.772588722239781, 2.7725887222397816, 2.7725887222397816, 2.7725887222397816, 0.0, 1.8325814637483102, 2.7725887222397816, 3.638316886832339, 3.638316886832339, 3.638316886832339, 0.0, 1.8325814637483102, 2.7725887222397816, 3.638316886832339, 4.50258359721299, 4.50258359721299, 0.0, 1.8325814637483102, 2.7725887222397816, 3.638316886832339, 4.50258359721299, 5.325175654050906]
# for i in range(len(brs)):
#     for j in range(len(brs)):
#         qoe_value = qoe(brs[i], brs[j])
#         qoe_values.append(qoe_value)
# print(qoe_values)

new_qoe_values = np.sort(qoe_values)
print(new_qoe_values)


[0.0, 1.83258146, 2.77258872, 3.63831689, 4.5025836, 5.32517565]
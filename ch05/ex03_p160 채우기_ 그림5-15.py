from ch05.ex01_basic_layer import *

p_apple, p_orrange = 100, 150
num_apple, num_orrange = 2, 3
tax = 1.1

App = MultiplyLayer()
p_A = App.forward(p_apple, num_apple)
Ora = MultiplyLayer()
p_O = Ora.forward(p_orrange, num_orrange)

Sum_fruit = AddLayer()
p_Tot = Sum_fruit.forward(p_A, p_O)

cal_Tax = MultiplyLayer()
p_aft_tax = cal_Tax.forward(p_Tot, tax)
print(p_aft_tax)


df_dtot, df_dtax = cal_Tax.backward(1)
print('df_dtot:', df_dtot)
print('df_dtax:', df_dtax)

df_dpA, df_dpO = Sum_fruit.backward(df_dtot)
print('df_dpA:', df_dpA)
print('df_dpO:', df_dpO)

df_dApp, df_dAppNum = App.backward(df_dpA)
df_dOra, df_dOraNum = Ora.backward(df_dpO)
print('df_dApp, df_dAppNum: ',df_dApp, df_dAppNum)
print('df_dOra, df_dOraNum: ',df_dOra, df_dOraNum)
#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


#como é composta a coluna Age
black_friday['Age'].unique()


# In[5]:


#como é a coluna Gender
black_friday['Gender'].unique()


# In[6]:


#filtra Gender igual a F e Age '26-35'
df_q2 = black_friday[(black_friday['Gender']=='F')&(black_friday['Age']=='26-35')]


# In[7]:


def q2():
    return df_q2['User_ID'].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[8]:


def q3():
    return int(black_friday['User_ID'].nunique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[9]:


def q4():
    return int(black_friday.dtypes.nunique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[10]:


# total de valores null
total_na = black_friday.isna().sum().sum()


# In[11]:


# temos que retirar as linhas com mais de 1 null
# apenas as colunas Product_Category_2 e Product_Category_3 tem valores nulos
# vamos contar quantas linhas é null na coluna Product_Category_2 E na Product_Category_3

duplicated_na_df = black_friday[(black_friday['Product_Category_2'].isna()) & (black_friday['Product_Category_3'].isna())]
duplicated_na = duplicated_na_df.shape[0]


# In[12]:


#% de linhas com null
# (total de nulos - nulos duplicados)/total de nulos
na_percentage = float((total_na-duplicated_na)/black_friday.shape[0])


# In[13]:


def q5():
    return na_percentage


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[14]:


def q6():
    return int(black_friday['Product_Category_3'].isna().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


# agrupando por Product_Category_3 e fazendo a contagem dos valores
# no final, ordena
df_q7 = black_friday.groupby('Product_Category_3')['Product_Category_3'].count().sort_values()


# In[16]:


def q7():
    return int(df_q7[df_q7 == df_q7.max()].index[0])


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[17]:


#normalizar min max
normalized_minmax=(black_friday['Purchase']-black_friday['Purchase'].min())/(black_friday['Purchase'].max()-black_friday['Purchase'].min())
black_friday['Purchase'] = normalized_minmax


# In[18]:


def q8():
    return float(normalized_minmax.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[19]:


#padronizar pela média
standardize_df=(black_friday['Purchase']-black_friday['Purchase'].mean())/black_friday['Purchase'].std()


# In[20]:


# verifica na série os valores do intervalo [-1,1]
# depois disso soma
df_q9 = standardize_df.apply(lambda x: (x>=-1) and (x<=1))


# In[21]:


def q9():
    return int(df_q9.sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[22]:


#vamos olhar o sub dataframe apenas com os Product_Category_2 null
prod_cat_2_na = black_friday[black_friday['Product_Category_2'].isna()]
cat_prod2_null = prod_cat_2_na.shape[0]


# In[23]:


#dentro deste sub dataframe, se o numero de null da coluna Product_Category_3 for igual ao
#numero de linhas do subframe, então é verdade que se Product_Category_2 for null, 
#Product_Category_3 também é null
cat_prod3_null = prod_cat_2_na[prod_cat_2_na['Product_Category_3'].isna()].shape[0]


# In[24]:


def q10():
    return cat_prod2_null == cat_prod3_null


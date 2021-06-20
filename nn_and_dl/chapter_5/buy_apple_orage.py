#!/usr/bin/python
# -*- coding: UTF-8 -*-
from layer_naive import *

# 自变量
## 苹果单价
app_unit_price = 100
## 苹果个数
app_num = 2

## 橘子单价
orange_unit_price = 150
## 橘子个数
origin_num = 3

## 消费税率
tax = 1.1

# 定义层
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 前向传播
apple_price = mul_apple_layer.forward(app_unit_price, app_num)
orange_price = mul_orange_layer.forward(orange_unit_price, origin_num)
fruit_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(fruit_price, tax)
print(price)

# 反向传播
dprice_price = 1
dprice_fruitPrice, dprice_tax = mul_tax_layer.backword(dprice_price)
dprice_applePrice, dprice_orangePrice = add_apple_orange_layer.backword(dprice_fruitPrice)
dprice_appleUnitPrice, dprice_appleNum = mul_apple_layer.backword(dprice_applePrice)
dprice_orangeUnitPrice, dprice_orangeNum = mul_orange_layer.backword(dprice_orangePrice)

print(dprice_appleNum, dprice_appleUnitPrice, dprice_orangeNum, dprice_orangeUnitPrice, dprice_tax)
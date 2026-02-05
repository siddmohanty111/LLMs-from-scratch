import stochastic_activations as StAF

strelu = StAF.StReLU()

print(strelu.forward(4, switch_proba=0))
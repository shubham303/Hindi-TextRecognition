from torchmetrics import CharErrorRate

preds = ["शुभम"]
target = ["शभम"]

metric = CharErrorRate()

print(metric(preds, target))

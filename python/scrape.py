from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

url = (
    "https://paperswithcode.com/sota/image-classification-on-cifar-10?dimension=PARAMS"
)

page = requests.get(url, timeout=10000).text
soup = BeautifulSoup(page, "html.parser")

# print(soup.prettify())

data = soup.find_all("script", {"type": "application/json"})
data = [x.text for x in data]
data_json = [json.loads(x) for x in data]
# print(json.dumps(data_json[2], indent=2, sort_keys=True))

print("keys", data_json[2][0].keys())

pd_data = {
    "model_name": [],
    "accuracy": [],
    "params": [],
}
for ele in data_json[2]:
    accuracy = 0.0
    if ele["raw_metrics"]["Accuracy"] is not None:
        accuracy = ele["raw_metrics"]["Accuracy"]
    elif ele["raw_metrics"]["Top-1 Accuracy"] is not None:
        accuracy = ele["raw_metrics"]["Top-1 Accuracy"]
    elif ele["raw_metrics"]["Percentage correct"] is not None:
        accuracy = ele["raw_metrics"]["Percentage correct"]

    params = 0
    if ele["raw_metrics"]["PARAMS"] is not None:
        params = ele["raw_metrics"]["PARAMS"]
    elif ele["raw_metrics"]["Parameters"] is not None:
        params = ele["raw_metrics"]["Parameters"]

    if accuracy != 0.0 and params != 0:
        pd_data["params"].append(params)
        pd_data["accuracy"].append(accuracy)
        pd_data["model_name"].append(ele["method_short"])

df = pd.DataFrame(pd_data)
print(pd_data["params"])
print(df.head(100))


plt.style.use("seaborn-v0_8-poster")

fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True)
ax.set_xlabel("Params")
ax.set_ylabel("Accuracy")
ax.set_title("CIFAR-10")
ax.scatter(df["params"], df["accuracy"], s=20, c="red", alpha=0.5)
ax.set_xscale("log")
plt.savefig("cifar10.png")

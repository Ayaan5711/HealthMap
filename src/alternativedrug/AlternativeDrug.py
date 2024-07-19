import pandas as pd 
import pickle


class AlternateDrug:

    # def __init__(self):
        

    # Load similarity-vector-data from pickle in the form of dictionary
    

    def recommendation(self,medicine):
        with open('src/models/medicine_dict.pkl', 'rb') as f:
            medicines_dict = pickle.load(f)

        medicines = pd.DataFrame(medicines_dict)
        with open('src/models/similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_medicines = []
        for i in medicines_list:
            recommended_medicines.append({
                'medicine_name': self.medicines.iloc[i[0]].Drug_Name,
                'pharmeasy_link': f"https://pharmeasy.in/search/all?name={medicines.iloc[i[0]].Drug_Name}"
            })
        return recommended_medicines 
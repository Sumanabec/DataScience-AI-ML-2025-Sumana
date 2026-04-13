import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np


def get_top_imp_features(model, preprocessor, top_count = None):
    feature_imp = model.feature_importances_
    feature_names =[name.split("__")[1] for name in preprocessor.get_feature_names_out()]
    feature_names = [name.split("_")[0] for name in feature_names]
    feature_imp_df = pd.DataFrame({
                                "Feature": feature_names, 
                                "importance_xgb": feature_imp
                                }).sort_values(by='importance_xgb', ascending= False)

    top_imp_features_df = feature_imp_df.iloc[:top_count, :]
    
    #print(top_imp_features_df.to_string())

    top_imp_features = top_imp_features_df["Feature"].to_list()
   
    return feature_imp_df


## load the model
model_file_path = os.path.join("models", "best_xgboost_model_with_30_features.joblib")
model = joblib.load(model_file_path)


# find top 10 features
top_30_features = get_top_imp_features(model["xgb"],  model["pre"], 30)
#print(top_30_features["Feature"].to_list())
# print(sum(top_30_features["importance_xgb"]))
#print((set(top_30_features["Feature"].to_list())))

# feature_names =[name.split("__")[1] for name in model["pre"].get_feature_names_out()]
# feature_names = [name.split("_")[0] for name in feature_names]
# print(set(feature_names))

st.title("House Price Prediction App")
st.markdown("---")
st.subheader("This app predicts the price of House")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns([4,1,4,1,4])


with col1:
    MSZoning = st.selectbox("Identifies the general zoning classification of the sale(MSZoning)", ['RL', 'RM', 'FV', 'RH', 'C (all)'])

with col3:
    GarageFinish = st.selectbox("Interior finish of the garage(GarageFinish)", ['Unf', 'RFn', 'Fin'])

with col5:
     Condition1 = st.selectbox("Proximity to various conditions(Condition1)", ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'])


col6, col7, col8, col9, col10 = st.columns([4,1,4,1,4])
with col6:
    ExterQual = st.selectbox("Evaluates the quality of the material on the exterior(ExterQual)", ['TA', 'Gd', 'Ex', 'Fa'])
with col8:
    Exterior2nd = st.selectbox("Exterior covering on house (Exterior2nd)", ['VinylSd','MetalSd','HdBoard','Wd Sdng','Plywood','CmentBd',
                                                                    'Wd Shng', 'Stucco','BrkFace','AsbShng','ImStucc','Brk Cmn',''
                                                                    'Stone','AsphShn','Other','CBlock'])

with col10:
    RoofStyle = st.selectbox("Type of roof(RoofStyle)", ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'])



col11, col12, col13, col14, col15 = st.columns([4,1,4,1,4])
with col11:
   YrSold = st.number_input("Year Sold (YYYY)", min_value=1800, step=1)

with col13:
   YearBuilt = st.number_input("Original construction year (YYYY)", min_value=1800, step=1)

with col15:
    YearRemodAdd = st.number_input("Remodel date (same as construction date if no remodeling or additions)", min_value=1800, step=1)

HouseAge = YrSold - YearBuilt
HouseRemodAge = YrSold - YearRemodAdd



col16, col17, col18, col19, col20 = st.columns([4,1,4,1,4])
with col16:
     LandContour = st.selectbox("Flatness of the property(LandContour)", ['Lvl', 'Bnk', 'HLS', 'Low'])

with col18:
    MSSubClass = st.selectbox("Identifies the type of dwelling involved in the sale(MSSubClass)", [20, 60, 50, 120, 30, 160, 70, 80, 90, 190, 85, 75, 45, 180, 40])
with col20:
    BsmtQual = st.selectbox("Evaluates the height of the basement(BsmtQual)", ['TA', 'Gd', 'Ex', 'Fa'])



col26, col27, col28, col29, col30 = st.columns([4,1,4,1,4])
with col26:
    KitchenQual = st.selectbox("Kitchen quality(KitchenQual)", ['TA', 'Gd', 'Ex', 'Fa'])
with col28:
    GarageCars = st.number_input("Size of garage in car capacity(GarageCars)", min_value=0, step=1)
with col30:
    PoolArea = st.number_input("Pool area in square feet(PoolArea)", min_value=0)



col31, col32, col33, col34, col35 = st.columns([4,1,4,1,4])
with col31:
    TotalBsmtSF = st.number_input("Total square feet of basement area(TotalBsmtSF)", min_value=0,)
with col33:
    firstFlrSF = st.number_input("First Floor square feet(firstFlrSF)", min_value=0)
with col35:
    secondFlrSF = st.number_input("Second floor square feet(secondFlrSF)", min_value=0)

TotalSF = firstFlrSF+secondFlrSF


col36, col37, col38, col39, col40  = st.columns([4,1,4,1,4])
with col36:
    FullBath = st.number_input("Full bathrooms above grade", min_value=0, step=1)
with col38:
    HalfBath = st.number_input("Half baths above grade", min_value=0, step=1)
with col40:
    BsmtFullBath = st.number_input("Basement full bathrooms", min_value=0, step=1)


col41, col42, col43, col44, col45 = st.columns([4,1,4,1,4])
with col41:
    BsmtHalfBath = st.number_input("Basement half bathrooms", min_value=0, step=1)
    TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath

with col43:
    HouseStyle = st.selectbox("Style of dwelling(HouseStyle)", ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'])

with col45:
    Exterior1st = st.selectbox("Exterior covering on house(Exterior1st)", ['VinylSd','HdBoard','MetalSd','Wd Sdng','Plywood','CemntBd','BrkFace',
                                                              'WdShing','Stucco','AsbShng','BrkComm','Stone','AsphShn','ImStucc','CBlock'])



col46, col47, col48, col49, col50 = st.columns([4,1,4,1,4])
with col46:
    KitchenAbvGr = st.number_input("Kitchen Above Ground(KitchenAbvGr)", min_value=0, step=1)

with col48:
    GarageType = st.selectbox("Garage location(GarageType)", ['Attchd', 'Detchd', 'BuiltIn', 'Basment', 'CarPort', '2Types'])

with col50:
    BldgType = st.selectbox("Type of dwelling(BldgType)", ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2fmCon'])
    

col51, col52, col53, col54, col55 = st.columns([4,1,4,1,4])
with col46:
    FireplaceQu = st.selectbox("Fireplace quality(FireplaceQu)", ['Select', 'Gd', 'TA', 'Fa', 'Ex', 'Po'])



with col48:
    Neighborhood = st.selectbox("Physical locations within Ames city limits(Neighborhood)", ['NAmes','CollgCr','OldTown','Edwards','Somerst','Gilbert',
                                                                            'NridgHt','Sawyer','NWAmes','SawyerW','BrkSide','Crawfor',
                                                                            'Mitchel','NoRidge','Timber','IDOTRR','ClearCr','StoneBr',
                                                                            'SWISU','MeadowV','Blmngtn','BrDale','Veenker','NPkVill','Blueste'])

with col50:
    OverallQual = st.selectbox("Rates the overall material and finish of the house(OverallQual)", [1,2,3,4,5,6,7,8,9,10])
    



col56, col57, col58, col59, col60 = st.columns([4,1,4,1,4])
with col56:
    CentralAir = st.selectbox("Central air conditioning(CentralAir)", ['Y', 'N'])
with col58:
    GarageCond = st.selectbox("Garage condition(GarageCond)", ['TA', 'Fa', 'Gd', 'Po', 'Ex'])
with col60:
    GrLivArea = st.number_input("Above grade (ground) living area square feet(GrLivArea)", min_value=0)

col61, col62, col63, col64, col605= st.columns([4,1,4,1,4])
with col61:
    Electrical = st.selectbox("Electrical system(Electrical)", ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])
with col63:
    BsmtFinSF1 = st.number_input("Type 1 finished square feet(BsmtFinSF1)", min_value=0)


submit = st.button("Predict")

st.markdown("""---""")



if submit:
    
    FireplaceQu = None if FireplaceQu == 'Select' else FireplaceQu

    columns = ['GarageCond', 'MSSubClass', 'MSZoning', 'Electrical', 'GarageFinish', 'KitchenAbvGr', 'BsmtFinSF1', 'CentralAir', 
                'Exterior1st', 'TotalBath', 'FireplaceQu', '2ndFlrSF', 'OverallQual', 'PoolArea', 'ExterQual', 'HouseStyle', 'Condition1', 
                'KitchenQual', 'GarageCars', 'GrLivArea', 'Exterior2nd', 'TotalSF', 'HouseAge', 'LandContour', 'HouseRemodAge', 
                'Neighborhood', 'BldgType', 'BsmtQual', 'RoofStyle', 'GarageType']
    
    values = np.array([[GarageCond, MSSubClass, MSZoning, Electrical, GarageFinish, KitchenAbvGr, BsmtFinSF1, CentralAir, 
                        Exterior1st, TotalBath, FireplaceQu, secondFlrSF, OverallQual, PoolArea, ExterQual, HouseStyle, Condition1, 
                        KitchenQual, GarageCars, GrLivArea, Exterior2nd, TotalSF, HouseAge, LandContour, HouseRemodAge, 
                        Neighborhood, BldgType, BsmtQual, RoofStyle, GarageType]])
   

    test_df = pd.DataFrame(values, columns=columns)

    print(test_df)

    prediction = model.predict(test_df)
    print(prediction)

    st.write(f"Predicted House Price: {prediction[0].round(2)}")
# import shap
# import numpy as np
# from data.split_data import split_data

# # Assuming `loaded_model` is your trained neural network model
# nn_model = loaded_model  

# # Select a small background dataset for SHAP explanations
# background = X_train[:25]

# def shap_explainer(model_type, model, background, X_train_wo):
#     if model_type == 'nn':
#         explainer = shap.KernelExplainer(model.predict, background)
#     elif model_type == 'rf':
#         explainer = shap.TreeExplainer(model)
#     else:
#         raise ValueError("Unsupported model type. Use 'nn' or 'rf'.")
    
#     # Compute SHAP values
#     shap_values = explainer.shap_values(background)

#     # Select an instance to visualize
#     instance_index = 5
#     rounded_shap_values = np.round(shap_values[instance_index], 1)
#     rounded_features = np.round(X_train_wo.iloc[instance_index], 1)
    
#     # Visualize results
#     shap.initjs()
#     shap.force_plot(explainer.expected_value, rounded_shap_values, rounded_features, matplotlib=True)

# # Call the function with 'nn' model type
# shap_explainer('nn', nn_model, background, X_train)

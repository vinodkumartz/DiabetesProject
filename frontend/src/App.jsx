import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";

const App = () => {
  const [formData, setFormData] = useState({
    race: "Caucasian",
    gender: "Female",
    age: 50,
    admission_type_id: 1,
    num_lab_procedures: 20,
    num_medications: 10,
    number_inpatient: 1,
    insulin: 0,
    change: 0,
    diabetesMed: 1,
  });

  
  const [error, setError] = useState(null);  // ✅ Define the error state
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState("");

  
  // const processFeatures = () => {
  //   return [
  //     formData.age,
  //     formData.admission_type_id,
  //     formData.num_lab_procedures,
  //     formData.num_medications,
  //     formData.number_inpatient,
  //     formData.insulin === "Up" ? 1 : 0,
  //     formData.change === "Ch" ? 1 : 0,
  //     formData.diabetesMed === "Yes" ? 1 : 0,
  //   ];
  // };


  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);  // Clear previous errors
    setPrediction(null);  // Reset previous prediction
    setExplanation(null);  // Reset explanation

    // Extract and process all features (Ensure 177 features are included)
    const features = [
        parseFloat(formData.age) || 0,
        parseFloat(formData.num_lab_procedures) || 0,
        parseFloat(formData.num_medications) || 0,
        parseFloat(formData.number_inpatient) || 0,
        parseFloat(formData.insulin === "Up" ? 1 : 0) || 0,
        parseFloat(formData.change === "Ch" ? 1 : 0) || 0,
        parseFloat(formData.diabetesMed === "Yes" ? 1 : 0) || 0,
        // Include all other required features here...
    ];

    console.log("Sending features:", features);
    console.log("Total features:", features.length);

    try {
        const response = await axios.post("http://127.0.0.1:5000/predict",
            { features: features },
            { headers: { "Content-Type": "application/json" } }
        );

        // ✅ Set Prediction & Explanation received from the backend
        setPrediction(response.data.prediction);
        setExplanation(response.data.explanation);

    } catch (err) {
        console.error("Error making prediction:", err.response ? err.response.data : err);
        setError("Failed to fetch prediction. Check backend logs.");
    }
  };

  

  return (
    <motion.div className="h-screen flex items-center justify-center bg-cover bg-center p-4"
      style={{ backgroundImage: "url('/img.png')" }}>
      <div className=" bg-emerald-300 p-7 rounded-lg shadow-lg w-full max-w-2xl h-auto min-h-[570px]">
        <h1 className="text-3xl font-bold text-center mb-6">Diabetes Readmission Prediction</h1>
  
        <label className="block mt-4 text-lg font-semibold">Age</label>
        <input
          type="range"
          name="age"
          min="0"
          max="100"
          value={formData.age}
          onChange={(e) => setFormData({ ...formData, age: Number(e.target.value) })}
          className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer"
        />
        <p className="text-lg">Selected Age: {formData.age}</p>
  
        <label className="block mt-4 text-lg font-semibold">Num Lab Procedures</label>
        <input
          type="number"
          name="num_lab_procedures"
          value={formData.num_lab_procedures}
          onChange={(e) => setFormData({ ...formData, num_lab_procedures: Number(e.target.value) })}
          className="w-full p-3 border rounded text-lg"
        />
  
        <label className="block mt-4 text-lg font-semibold">Num Medications</label>
        <input
          type="number"
          name="num_medications"
          value={formData.num_medications}
          onChange={(e) => setFormData({ ...formData, num_medications: Number(e.target.value) })}
          className="w-full p-3 border rounded text-lg"
        />
  
        <label className="block mt-4 text-lg font-semibold">Number of Inpatient Visits</label>
        <input
          type="number"
          name="number_inpatient"
          value={formData.number_inpatient}
          onChange={(e) => setFormData({ ...formData, number_inpatient: Number(e.target.value) })}
          className="w-full p-3 border rounded text-lg"
        />
  
        <label className="block mt-4 text-lg font-semibold">Insulin</label>
        <select
          name="insulin"
          onChange={(e) => setFormData({ ...formData, insulin: e.target.value })}
          className="w-full p-3 border rounded bg-gray-100 text-lg"
        >
          <option value="No">No</option>
          <option value="Up">Up</option>
          <option value="Down">Down</option>
        </select>
  
        <label className="block mt-4 text-lg font-semibold">Change</label>
        <select
          name="change"
          onChange={(e) => setFormData({ ...formData, change: e.target.value })}
          className="w-full p-3 border rounded bg-gray-100 text-lg"
        >
          <option value="No">No</option>
          <option value="Ch">Ch</option>
        </select>
  
        <label className="block mt-4 text-lg font-semibold">Diabetes Medication</label>
        <select
          name="diabetesMed"
          onChange={(e) => setFormData({ ...formData, diabetesMed: e.target.value })}
          className="w-full p-3 border rounded bg-gray-100 text-lg"
        >
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
  
        <button
          onClick={handleSubmit}
          className="mt-6 bg-blue-500 text-brown font-bold px-6 py-3 rounded text-lg font-semibold w-full"
        >
          Predict
        </button>
  
        {error && <p className="text-red-500 mt-4 text-lg">{error}</p>}
  
        {prediction && (
          <motion.div className="mt-6 p-5 border rounded bg-gray-100 shadow-lg">
            <p className="text-xl font-bold text-green-700">Prediction: {prediction}</p>
            <p className="text-lg text-gray-700 mt-2">
              Explanation: {explanation || "No explanation available."}
            </p>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default App;
// frontend/src/components/PredictionForm.js

import React, { useState } from 'react';
import axios from 'axios';

const PredictionForm = () => {
    const [formData, setFormData] = useState({
        state: '',
        bedrooms: 1,
        baths: 1,
        salary_per_year: ''
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        // Prepare data
        const payload = {
            state: formData.state,
            bedrooms: parseInt(formData.bedrooms),
            baths: parseInt(formData.baths),
        };

        if (formData.salary_per_year.trim() !== '') {
            payload.salary_per_year = parseFloat(formData.salary_per_year);
        }

        try {
            const response = await axios.post('http://127.0.0.1:8000/predict', payload);
            setResult(response.data);
        } catch (err) {
            if (err.response) {
                setError(err.response.data.detail);
            } else {
                setError('An unexpected error occurred.');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="prediction-form">
            <h2>House Price Prediction and Affordability Checker</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>State:</label>
                    <input
                        type="text"
                        name="state"
                        value={formData.state}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Number of Bedrooms:</label>
                    <input
                        type="number"
                        name="bedrooms"
                        min="1"
                        value={formData.bedrooms}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Number of Bathrooms:</label>
                    <input
                        type="number"
                        name="baths"
                        min="1"
                        value={formData.baths}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Yearly Salary (optional):</label>
                    <input
                        type="number"
                        name="salary_per_year"
                        min="0"
                        value={formData.salary_per_year}
                        onChange={handleChange}
                        placeholder="Enter your yearly salary"
                    />
                </div>
                <button type="submit" disabled={loading}>
                    {loading ? 'Predicting...' : 'Predict'}
                </button>
            </form>

            {error && <div className="error">Error: {error}</div>}

            {result && (
                <div className="result">
                    <h3>Prediction Results:</h3>
                    <p><strong>Predicted Price:</strong> ${result.predicted_price.toLocaleString()}</p>
                    {result.yearly_income && (
                        <>
                            <p><strong>Yearly Income:</strong> ${result.yearly_income.toLocaleString()}</p>
                            <p><strong>Monthly Payment:</strong> ${result.monthly_payment.toLocaleString(undefined, {minimumFractionDigits: 2})}</p>
                            <p><strong>Affordability Status:</strong> {result.affordability_status}</p>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

export default PredictionForm;

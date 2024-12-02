document.getElementById('submitBtn').addEventListener('click', async () => {
    const formData = {
        sex: document.getElementById('sex').value,
        age: parseInt(document.getElementById('age').value),
        Pstatus: document.getElementById('Pstatus').value,
        Medu: parseInt(document.getElementById('Medu').value),
        Fedu: parseInt(document.getElementById('Fedu').value),
        reason: document.getElementById('reason').value,
        failures: parseInt(document.getElementById('failures').value),
        schoolsup: document.getElementById('schoolsup').value,
        activities: document.getElementById('activities').value,
        nursery: document.getElementById('nursery').value,
        higher: document.getElementById('higher').value,
        internet: document.getElementById('internet').value,
        romantic: document.getElementById('romantic').value,
        famrel: parseInt(document.getElementById('famrel').value),
        goout: parseInt(document.getElementById('goout').value),
        Dalc: parseInt(document.getElementById('Dalc').value),
        Walc: parseInt(document.getElementById('Walc').value),
        health: parseInt(document.getElementById('health').value),
        absences: parseInt(document.getElementById('absences').value),
        G1: parseInt(document.getElementById('G1').value),
        G2: parseInt(document.getElementById('G2').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        const finalGrade = parseFloat(result[0]).toFixed(2);
        document.getElementById('resultBox').value = `A predição calculou a nota final como: ${finalGrade}`;
    } catch (error) {
        document.getElementById('resultBox').value = 'Error: ' + error.message;
    }
});

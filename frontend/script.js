const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const predictionList = document.getElementById('prediction-list');
const predictBtn = document.getElementById('predict-btn');

fileInput.addEventListener('change', () => {
  const selected = fileInput.files[0];
  fileName.textContent = selected ? selected.name : 'Choose an image';
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = 'Please choose an image file.';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  statusEl.textContent = 'Predicting...';
  resultsEl.classList.add('hidden');
  predictionList.innerHTML = '';
  predictBtn.disabled = true;

  try {
    const response = await fetch('/api/predict?top_k=3', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Prediction failed.');
    }

    for (const item of data.predictions) {
      const li = document.createElement('li');
      const font = document.createElement('span');
      font.textContent = item.font;
      const confidence = document.createElement('span');
      confidence.textContent = `${(item.confidence * 100).toFixed(2)}%`;
      li.appendChild(font);
      li.appendChild(confidence);
      predictionList.appendChild(li);
    }

    resultsEl.classList.remove('hidden');
    statusEl.textContent = 'Prediction complete.';
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
  }
});

const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const predictionList = document.getElementById('prediction-list');
const predictBtn = document.getElementById('predict-btn');

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.classList.toggle('error', isError);
};

fileInput.addEventListener('change', () => {
  const selected = fileInput.files[0];
  fileName.textContent = selected ? selected.name : 'No file selected';
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    setStatus('Please select an image first.', true);
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  setStatus('Running prediction...');
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

    data.predictions.forEach((item) => {
      const row = document.createElement('li');
      const font = document.createElement('strong');
      font.textContent = item.font;
      const confidence = document.createElement('span');
      confidence.textContent = `${(item.confidence * 100).toFixed(2)}%`;
      row.appendChild(font);
      row.appendChild(confidence);
      predictionList.appendChild(row);
    });

    resultsEl.classList.remove('hidden');
    setStatus('Done.');
  } catch (error) {
    setStatus(error.message || 'Something went wrong.', true);
  } finally {
    predictBtn.disabled = false;
  }
});

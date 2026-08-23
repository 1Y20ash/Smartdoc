const fileInput = document.getElementById('document');
const uploadZone = document.querySelector('.upload-zone');
const filename = document.getElementById('filename');
const dropTitle = document.getElementById('dropTitle');
const form = document.getElementById('uploadForm');
const analyzeButton = document.getElementById('analyzeButton');

function showFile(file) {
  if (!file) return;
  filename.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  dropTitle.textContent = 'Document ready to analyze';
  uploadZone.classList.add('file-selected');
}

fileInput?.addEventListener('change', () => showFile(fileInput.files[0]));

['dragenter', 'dragover'].forEach(eventName => {
  uploadZone?.addEventListener(eventName, event => {
    event.preventDefault();
    uploadZone.classList.add('is-dragging');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  uploadZone?.addEventListener(eventName, event => {
    event.preventDefault();
    uploadZone.classList.remove('is-dragging');
  });
});

uploadZone?.addEventListener('drop', event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const allowed = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
  const extensionAllowed = /\.(pdf|docx|txt)$/i.test(file.name);
  if (!allowed.includes(file.type) && !extensionAllowed) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showFile(file);
});

form?.addEventListener('submit', () => {
  analyzeButton.disabled = true;
  analyzeButton.innerHTML = 'Analyzing document <b>…</b>';
});

document.addEventListener('DOMContentLoaded', () => {
    const ingestForm = document.getElementById('ingest-form');
    const fileInput = document.getElementById('file-input');
    const ingestionStatus = document.getElementById('ingestion-status');

    ingestForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            ingestionStatus.textContent = "Please select a file to upload.";
            ingestionStatus.style.color = 'red';
            return;
        }

        ingestionStatus.textContent = "Uploading and ingesting file...";
        ingestionStatus.style.color = 'yellow';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:8000/ingest', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            ingestionStatus.textContent = data.message;
            ingestionStatus.style.color = 'lightgreen';
            console.log("Ingestion successful:", data);
        } catch (error) {
            console.error("Ingestion failed:", error);
            ingestionStatus.textContent = `Ingestion failed: ${error.message}`;
            ingestionStatus.style.color = 'red';
        }
    });
});
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the form
    const analysisForm = document.getElementById('analysisForm');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            e.preventDefault();
            analyzeStock();
        });
    }
    
    // Load popular stocks
    loadPopularStocks();
});

function loadPopularStocks() {
    // You can populate a dropdown with popular stocks here
    fetch('/api/stocks')
        .then(response => response.json())
        .then(data => {
            console.log('Available stocks:', data.stocks);
        })
        .catch(error => {
            console.error('Error loading stocks:', error);
        });
}

function analyzeStock() {
    const symbol = document.getElementById('symbol').value.toUpperCase();
    const period = document.getElementById('period').value;
    
    if (!symbol) {
        showError('Please enter a stock symbol');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    const resultsDiv = document.getElementById('results');
    if (resultsDiv) {
        resultsDiv.style.display = 'none';
    }
    
    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `symbol=${symbol}&period=${period}`
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
            // Redirect to analysis page
            window.location.href = `/analysis?symbol=${symbol}&period=${period}`;
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        showError('An error occurred while analyzing the stock: ' + error.message);
        console.error('Error:', error);
    });
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '';
    
    // Basic results display for home page
    const resultsHtml = `
        <div class="alert alert-success">
            <h5>Analysis Complete!</h5>
            <p>Successfully analyzed ${data.symbol}. Redirecting to detailed analysis...</p>
        </div>
    `;
    
    resultsDiv.innerHTML = resultsHtml;
    resultsDiv.style.display = 'block';
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    resultsDiv.style.display = 'block';
}
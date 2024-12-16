/**
 * @jest-environment jsdom
 */

describe('Lyrics Generator Frontend Tests', () => {
    beforeEach(() => {
        // Setup DOM
        document.body.innerHTML = `
            <input id="artist" type="text">
            <select id="genre">
                <option value="pop">Pop</option>
                <option value="rock">Rock</option>
            </select>
            <input id="maxLength" type="number" value="200">
            <input id="temperature" type="range" value="1.0">
            <button id="generateBtn">Generate</button>
            <div id="lyricsOutput"></div>
            <div class="loading"></div>
        `;

        // Mock fetch
        global.fetch = jest.fn();
        
        // Add event listeners
        const generateBtn = document.getElementById('generateBtn');
        const loading = document.querySelector('.loading');
        const output = document.getElementById('lyricsOutput');
        
        generateBtn.addEventListener('click', async function() {
            const artist = document.getElementById('artist').value;
            const genre = document.getElementById('genre').value;
            const maxLength = parseInt(document.getElementById('maxLength').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            loading.style.display = 'block';
            this.disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        artist,
                        genre,
                        max_length: maxLength,
                        temperature
                    })
                });
                
                const data = await response.json();
                if (data.error) {
                    output.textContent = 'Error: ' + data.error;
                } else {
                    output.textContent = data.lyrics;
                }
            } catch (error) {
                output.textContent = 'Error generating lyrics. Please try again.';
            } finally {
                loading.style.display = 'none';
                this.disabled = false;
            }
        });
    });

    afterEach(() => {
        jest.resetAllMocks();
    });

    test('UI elements exist', () => {
        expect(document.getElementById('artist')).toBeTruthy();
        expect(document.getElementById('genre')).toBeTruthy();
        expect(document.getElementById('maxLength')).toBeTruthy();
        expect(document.getElementById('temperature')).toBeTruthy();
        expect(document.getElementById('generateBtn')).toBeTruthy();
        expect(document.getElementById('lyricsOutput')).toBeTruthy();
    });

    test('Generate button triggers API call with correct data', async () => {
        // Setup test data
        document.getElementById('artist').value = 'Test Artist';
        document.getElementById('genre').value = 'pop';
        document.getElementById('maxLength').value = '150';
        document.getElementById('temperature').value = '0.8';

        // Mock successful API response
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ lyrics: 'Test generated lyrics' })
        });

        // Trigger generate button click
        document.getElementById('generateBtn').click();
        
        // Wait for async operations
        await new Promise(resolve => setTimeout(resolve, 100));

        // Check if fetch was called with correct data
        expect(fetch).toHaveBeenCalledWith('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                artist: 'Test Artist',
                genre: 'pop',
                max_length: 150,
                temperature: 0.8
            })
        });
    });

    test('Loading state is handled correctly', async () => {
        // Mock delayed API response
        global.fetch.mockImplementationOnce(() => 
            new Promise(resolve => 
                setTimeout(() => resolve({
                    ok: true,
                    json: async () => ({ lyrics: 'Test lyrics' })
                }), 100)
            )
        );

        // Get elements
        const loading = document.querySelector('.loading');
        const button = document.getElementById('generateBtn');

        // Initial state
        expect(loading.style.display).toBe('');
        expect(button.disabled).toBeFalsy();

        // Trigger generation
        button.click();
        
        // Wait for loading state
        await new Promise(resolve => setTimeout(resolve, 50));

        // Check loading state
        expect(loading.style.display).toBe('block');
        expect(button.disabled).toBeTruthy();

        // Wait for response
        await new Promise(resolve => setTimeout(resolve, 150));

        // Check final state
        expect(loading.style.display).toBe('none');
        expect(button.disabled).toBeFalsy();
    });

    test('Error handling', async () => {
        // Mock API error
        global.fetch.mockRejectedValueOnce(new Error('API Error'));

        // Get output element
        const output = document.getElementById('lyricsOutput');

        // Trigger generation
        document.getElementById('generateBtn').click();

        // Wait for error handling
        await new Promise(resolve => setTimeout(resolve, 100));

        // Check error message
        expect(output.textContent).toContain('Error');
    });
});

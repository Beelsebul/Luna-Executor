document.addEventListener("DOMContentLoaded", function () {
    const gameBoard = document.getElementById('gameBoard');
    const rows = 13;
    const cols = 13;
    const hexWidth = 40; // Width of a hexagon
    const marginIncrement = hexWidth * 0.5 + 2; // marginIncrement = hexWidth * sqrt(3)/2

    gameBoard.style.setProperty('--hex-width', `${hexWidth}px`);
    gameBoard.style.setProperty('margin-top', `100px`);

    let letters = 'ABCDEFGHIJK';
    let moves = [];
    let currentMoveIndex = 0;
    let inputMoves = [];
    const uniqueMoves = new Set();
    let isGameOver = false; // Флаг завершения игры

    // Создаем игровое поле 11x11
    const boardSize = 11;
    const board = Array.from(Array(boardSize), () => Array(boardSize).fill(0));

    // Генерация игрового поля
    for (let i = 0; i < rows; i++) {
        const row = document.createElement('div');
        row.classList.add('hex-row');
        row.style.marginLeft = i * marginIncrement + 'px';
        for (let j = 0; j < cols; j++) {
            let hex;
            if ((i === 0 && j === 12)) {
                hex = document.createElement('div');
                hex.classList.add('hex_special1');
            } else if (i === 12 && j === 0) {
                hex = document.createElement('div');
                hex.classList.add('hex_special2');
            } else {
                hex = document.createElement('div');
                hex.classList.add('hexagon');
                if (
                    (i === 0 && j === 0) || 
                    (i === 0 && j === cols - 1) || 
                    (i === rows - 1 && j === 0) || 
                    (i === rows - 1 && j === cols - 1)
                ) {
                    hex.classList.add('hex_corner');
                } else if (i === 0 || i === rows - 1) {
                    hex.classList.add('hex_border', 'horizontal');
                } else if (j === 0 || j === cols - 1) {
                    hex.classList.add('hex_border', 'vertical');
                } else {
                    hex.classList.add('hex_playable');
                    if (i !== 0 && i !== rows - 1 && j !== 0 && j !== cols - 1) {
                        const letter = letters[j - 1];
                        const number = i; // Keep original index logic
                        hex.id = `${letter}${number}`;
                    }
                }
            }
            row.appendChild(hex);
        }
        gameBoard.appendChild(row);
    }

    const textInput = document.getElementById('textInput');
    const hoverInfo = document.getElementById('hoverInfo');
    const gameLog = document.getElementById('gameLog');
    const currentMoveDisplay = document.getElementById('currentMove');
    const postButton = document.getElementById('postButton');
    const getButton = document.getElementById('getButton');

    textInput.addEventListener('input', function (e) {
        // Разрешаем только допустимые идентификаторы клеток
        textInput.value = textInput.value.replace(/[^A-Ka-k0-9 ]/g, '').replace(/\s{2,}/g, ' ').toUpperCase();
    });

    postButton.addEventListener('click', function () {
        inputMoves = textInput.value.trim().split(' ');
        applyMoves();
    });

    getButton.addEventListener('click', function () {
        textInput.value = moves.join(' ');
    });

    document.querySelectorAll('.hex_playable').forEach(hex => {
        let timeout;
        hex.addEventListener('mouseenter', function () {
            timeout = setTimeout(() => {
                hoverInfo.innerText = hex.id;
                hoverInfo.classList.add('show');
            }, 10);
        });
        hex.addEventListener('mouseleave', function () {
            clearTimeout(timeout);
            hoverInfo.classList.remove('show');
        });
        hex.addEventListener('click', function () {
            if (isGameOver) return; // Блокируем ход, если игра окончена
            const move = hex.id;
            if (!uniqueMoves.has(move)) {
                uniqueMoves.add(move);
                addMove(move);
                displayLog();
                displayBoardState(moves.length);
                checkVictory();
                checkAITurn();  // Проверяем, если AI должен сгенерировать ход
            }
        });
    });

    function applyMoves() {
        moves = [];
        uniqueMoves.clear();
        resetBoard(); // Сбрасываем доску перед применением ходов
        for (const move of inputMoves) {
            if (!isValidMove(move)) break;
            if (!uniqueMoves.has(move)) {
                uniqueMoves.add(move);
                moves.push(move);
            }
        }
        applyMovesToBoard(moves); // Применяем ходы к доске
        displayLog();
        displayBoardState(moves.length);
        checkVictory();
    }

    function isValidMove(move, size = 11) {
        if (move.length < 2) {
            return false;
        }
        const letter = move[0].toUpperCase();
        if (letter < 'A' || letter > String.fromCharCode('A'.charCodeAt(0) + size - 1)) {
            return false;
        }
        const number = parseInt(move.slice(1));
        return !isNaN(number) && number >= 1 && number <= size;
    }

    function addMove(move) {
        moves = moves.slice(0, currentMoveIndex);
        moves.push(move);
        uniqueMoves.clear();
        for (let i = 0; i < moves.length; i++) {
            uniqueMoves.add(moves[i]);
        }
        applyMovesToBoard(moves); // Применяем ходы к доске
        displayLog();
        displayBoardState(moves.length);
        checkVictory();
    }

    function displayLog() {
        gameLog.innerHTML = '';
        moves.forEach((move, index) => {
            const span = document.createElement('span');
            span.textContent = move + ' ';
            if (index === currentMoveIndex - 1) {
                span.style.color = 'blue';
            }
            gameLog.appendChild(span);
        });
    }

    function displayBoardState(moveIndex) {
        document.querySelectorAll('.hex_playable').forEach(hex => {
            hex.classList.remove('red', 'blue');
        });

        for (let i = 0; i < moveIndex; i++) {
            const hex = document.getElementById(moves[i]);
            if (hex) {
                if (!hex.classList.contains('red') && !hex.classList.contains('blue')) {
                    if (i % 2 === 0) {
                        hex.classList.add('red');
                    } else {
                        hex.classList.add('blue');
                    }
                }
            }
        }
        currentMoveDisplay.textContent = moveIndex;
        currentMoveIndex = moveIndex;
        displayLog();
        checkVictory();
    }

    document.getElementById('prevMove').addEventListener('click', function () {
        if (currentMoveIndex > 0) {
            isGameOver = false; // Снимаем ограничение
            currentMoveIndex--;
            moves.pop(); // Удаляем последний ход
            uniqueMoves.delete(moves[moves.length - 1]);
            applyMovesToBoard(moves); // Обновляем доску
            displayBoardState(currentMoveIndex);
        }
    });

    document.getElementById('nextMove').addEventListener('click', function () {
        // Не реализовано, так как нет сохранённых будущих ходов
    });

    function resetBoard() {
        for (let i = 0; i < boardSize; i++) {
            for (let j = 0; j < boardSize; j++) {
                board[i][j] = 0;
            }
        }
    }

    function applyMovesToBoard(moves) {
        resetBoard();
        let appliedMoves = [];
        const _allow_swap_rule = true;

        for (let i = 0; i < moves.length; i++) {
            const move = moves[i];
            if (!isValidMove(move)) {
                console.error('Invalid move:', move);
                return;
            }
            let [x, y] = convertMoveToCoords(move);

            if (i === 1 && _allow_swap_rule) {
                const firstMove = moves[0];
                if (move === firstMove) {
                    // Swap rule
                    const [fx, fy] = convertMoveToCoords(firstMove);
                    board[fy][fx] = 0; // Удаляем ход красного
                    [x, y] = [y, x]; // Меняем координаты местами
                    const swapMove = coordsToMove(x, y);
                    board[y][x] = 2; // Ставим синий камень
                    appliedMoves = [[x, y]];
                } else {
                    board[y][x] = 2; // Обычный ход синего
                    appliedMoves.push([x, y]);
                }
            } else {
                if (board[y][x] !== 0) {
                    console.error('Cell is already occupied:', move);
                    return;
                }
                board[y][x] = (i % 2 === 0) ? 1 : 2;
                appliedMoves.push([x, y]);
            }
        }
    }

    function convertMoveToCoords(move) {
        const letter = move[0].toUpperCase();
        const number = parseInt(move.slice(1)) - 1;
        const x = letter.charCodeAt(0) - 'A'.charCodeAt(0);
        const y = number;
        return [x, y];
    }

    function coordsToMove(x, y) {
        const letter = String.fromCharCode(x + 'A'.charCodeAt(0));
        const number = y + 1;
        return `${letter}${number}`;
    }

    function getNeighbors(i, j, size = boardSize) {
        const directions = [
            [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, -1]
        ];
        return directions
            .map(([di, dj]) => [i + di, j + dj])
            .filter(([ni, nj]) => ni >= 0 && ni < size && nj >= 0 && nj < size);
    }

    function dfs(board, start, player, goalEdge) {
        const size = boardSize;
        const stack = [start];
        const visited = new Set();

        while (stack.length > 0) {
            const current = stack.pop();
            const key = `${current[0]},${current[1]}`;
            if (!visited.has(key)) {
                visited.add(key);
                const [i, j] = current;

                if (goalEdge(i, j)) {
                    return true;
                }

                const neighbors = getNeighbors(i, j, size);
                for (const [ni, nj] of neighbors) {
                    if (board[ni][nj] === player && !visited.has(`${ni},${nj}`)) {
                        stack.push([ni, nj]);
                    }
                }
            }
        }
        return false;
    }

    function checkWinnerRed(board) {  // Red (1) соединяет верх и низ
        const size = boardSize;
        for (let j = 0; j < size; j++) {
            if (board[0][j] === 1) {  // Проверяем верхний край
                if (dfs(board, [0, j], 1, (i, j) => i === size - 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    function checkWinnerBlue(board) {  // Blue (2) соединяет левую и правую стороны
        const size = boardSize;
        for (let i = 0; i < size; i++) {
            if (board[i][0] === 2) {  // Проверяем левый край
                if (dfs(board, [i, 0], 2, (i, j) => j === size - 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    function checkVictory() {
        if (checkWinnerRed(board)) {
            showVictoryMessage("Red wins!", "red");
            return;
        }
        if (checkWinnerBlue(board)) {
            showVictoryMessage("Blue wins!", "blue");
            return;
        }
    }

    function showVictoryMessage(message, color) {
        // Проверяем, есть ли уже сообщение о победе
        if (document.querySelector('.victory-message')) {
            return; // Сообщение уже есть, не создаём новое
        }

        isGameOver = true;

        // Создаем элемент сообщения
        const victoryPopup = document.createElement('div');
        victoryPopup.classList.add('victory-message');
        victoryPopup.style.backgroundColor = color === "red" ? "#F65252" : "#52BDF6";

        const messageText = document.createElement('p');
        messageText.textContent = message;
        victoryPopup.appendChild(messageText);

        // Кнопка закрытия
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.classList.add('close-button');
        closeButton.onclick = () => {
            hideVictoryMessage(victoryPopup); // Плавно скрываем сообщение
        };
        victoryPopup.appendChild(closeButton);

        // Кнопка новой игры
        const newGameButton = document.createElement('button');
        newGameButton.textContent = 'New Game';
        newGameButton.classList.add('new-game-button');
        newGameButton.onclick = () => {
            startNewGame(); // Начинаем новую игру
            hideVictoryMessage(victoryPopup); // Скрываем сообщение
        };
        victoryPopup.appendChild(newGameButton);

        // Добавляем сообщение на страницу
        document.body.appendChild(victoryPopup);

        // Плавное появление через добавление класса
        setTimeout(() => {
            victoryPopup.classList.add('show');
        }, 10);
    }

    // Функция для скрытия сообщения с удалением
    function hideVictoryMessage(victoryPopup) {
        // Убираем класс, чтобы началась анимация исчезновения
        victoryPopup.classList.remove('show');

        // Добавляем обработчик события окончания перехода
        victoryPopup.addEventListener('transitionend', function(event) {
            // Проверяем, что завершился переход для свойства 'opacity' или 'transform'
            if (event.propertyName === 'opacity' || 'transform') {
                if (victoryPopup && victoryPopup.parentNode) {
                    victoryPopup.parentNode.removeChild(victoryPopup); // Удаляем элемент из DOM
                }
            }
        }, { once: true }); // Событие срабатывает только один раз
    }

    function startNewGame() {
        isGameOver = false; // Снимаем ограничение
        moves = [];
        uniqueMoves.clear();
        currentMoveIndex = 0;
        resetBoard();
        displayBoardState(0);
        displayLog();
    }

    document.getElementById('newGameButton').addEventListener('click', function () {
        startNewGame();
    });

    // ---------- Новый код для AI ----------

    // Функция для отправки текущих ходов на сервер и получения ответа AI
   
    function fetchNextMoveAI() {
        const currentTurn = moves.length % 2 === 0 ? 'red' : 'blue';

        fetch('/next-move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                moves: moves.join(' '),
                player: currentTurn // Specify which player (red or blue) is making the move
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.next_moves) {
                let nextMove = null;

                for (const move of data.next_moves) {
                    if (move === "<endoftext>" || move === "<endofgame>") {
                        continue;
                    }

                    if (isValidMove(move) && !isCellOccupied(move)) {
                        nextMove = move;
                        break;
                    }
                }

                if (nextMove) {
                    addMove(nextMove); // Add the valid move to the board
                    displayLog();
                    displayBoardState(moves.length);
                    checkVictory();

                    // Check if AI should make the next move
                    setTimeout(checkAITurn, 500); // Add a delay for better UX
                } else {
                    console.error('No valid moves found.');
                }
            } else {
                console.error('Error generating move:', data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    function checkAITurn() {
        if (isGameOver) return; // Stop if the game has ended

        const currentTurn = moves.length % 2 === 0 ? 'red' : 'blue';

        // Trigger AI move if the current player has AI enabled
        if ((currentTurn === 'blue' && aiBluePlaysToggle.checked) ||
            (currentTurn === 'red' && aiRedPlaysToggle.checked)) {
            fetchNextMoveAI();
        }
    }

    function isCellOccupied(move) {
        const [x, y] = convertMoveToCoords(move);
        return board[y][x] !== 0; // Returns true if the cell is already occupied
    }
    // Start game with AI move if enabled
    document.getElementById('newGameButton').addEventListener('click', function () {
        startNewGame();
        checkAITurn(); // Trigger AI move if an AI player is enabled at game start
    });

    // Initialize toggles to trigger AI check without needing button press
    const aiBluePlaysToggle = document.getElementById('aiBluePlaysToggle');
    const aiRedPlaysToggle = document.getElementById('aiRedPlaysToggle');

    aiBluePlaysToggle.addEventListener('change', checkAITurn);
    aiRedPlaysToggle.addEventListener('change', checkAITurn);
});

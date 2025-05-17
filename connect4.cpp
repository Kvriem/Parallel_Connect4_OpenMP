#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <iomanip>
#include <windows.h>
#include <string>
#include <cstdlib>
#include <ctime>

const int DEPTH = 4;
const int ROWS = 6;
const int COLUMNS = 7;
const int EMPTY = 0;
const int HUMAN = 1;
const int AI = 2;
const int MAX_SPACE_TO_WIN = 3;

using Board = std::vector<std::vector<int>>;

Board createBoard() {
    return Board(ROWS, std::vector<int>(COLUMNS, EMPTY));
}

bool isValidColumn(const Board& board, int column) {
    return board[0][column - 1] == EMPTY;
}

std::vector<int> validLocations(const Board& board) {
    std::vector<int> valid;
    #pragma omp parallel for
    for (int i = 1; i <= 7; i++) {
        if (isValidColumn(board, i)) {
            #pragma omp critical
            valid.push_back(i);
        }
    }
    return valid;
}

void placePiece(Board& board, int player, int column) {
    int index = column - 1;
    for (int row = ROWS - 1; row >= 0; row--) {
        if (board[row][index] == EMPTY) {
            board[row][index] = player;
            return;
        }
    }
}

Board cloneAndPlacePiece(const Board& board, int player, int column) {
    Board newBoard = board;
    placePiece(newBoard, player, column);
    return newBoard;
}

bool detectWin(const Board& board, int player) {
    bool found = false;
    
    // Horizontal win
    #pragma omp parallel for collapse(2) shared(found)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = 0; row < ROWS; row++) {
            if (!found && board[row][col] == player && 
                board[row][col+1] == player && 
                board[row][col+2] == player && 
                board[row][col+3] == player) {
                #pragma omp critical
                found = true;
            }
        }
    }
    if (found) return true;

    // Vertical win
    #pragma omp parallel for collapse(2) shared(found)
    for (int col = 0; col < COLUMNS; col++) {
        for (int row = 0; row < ROWS - MAX_SPACE_TO_WIN; row++) {
            if (!found && board[row][col] == player && 
                board[row+1][col] == player && 
                board[row+2][col] == player && 
                board[row+3][col] == player) {
                #pragma omp critical
                found = true;
            }
        }
    }
    if (found) return true;

    // Diagonal upwards win
    #pragma omp parallel for collapse(2) shared(found)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = 0; row < ROWS - MAX_SPACE_TO_WIN; row++) {
            if (!found && board[row][col] == player && 
                board[row+1][col+1] == player && 
                board[row+2][col+2] == player && 
                board[row+3][col+3] == player) {
                #pragma omp critical
                found = true;
            }
        }
    }
    if (found) return true;

    // Diagonal downwards win
    #pragma omp parallel for collapse(2) shared(found)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = MAX_SPACE_TO_WIN; row < ROWS; row++) {
            if (!found && board[row][col] == player && 
                board[row-1][col+1] == player && 
                board[row-2][col+2] == player && 
                board[row-3][col+3] == player) {
                #pragma omp critical
                found = true;
            }
        }
    }
    return found;
}

bool isTerminalBoard(const Board& board) {
    return detectWin(board, HUMAN) || detectWin(board, AI) || validLocations(board).empty();
}

int evaluateAdjacents(const std::vector<int>& adjacentPieces, int player) {
    int opponent = (player == AI) ? HUMAN : AI;
    int score = 0;
    int playerPieces = 0;
    int emptySpaces = 0;
    int opponentPieces = 0;

    for (int p : adjacentPieces) {
        if (p == player) playerPieces++;
        else if (p == EMPTY) emptySpaces++;
        else if (p == opponent) opponentPieces++;
    }

    if (playerPieces == 4) score += 99999;
    else if (playerPieces == 3 && emptySpaces == 1) score += 100;
    else if (playerPieces == 2 && emptySpaces == 2) score += 10;

    return score;
}

int score(const Board& board, int player) {
    int totalScore = 0;
    
    // Center column bonus
    #pragma omp parallel for reduction(+:totalScore)
    for (int col = 2; col < 5; col++) {
        for (int row = 0; row < ROWS; row++) {
            if (board[row][col] == player) {
                totalScore += (col == 3) ? 3 : 2;
            }
        }
    }

    // Horizontal evaluation
    #pragma omp parallel for reduction(+:totalScore)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = 0; row < ROWS; row++) {
            std::vector<int> adjacentPieces = {
                board[row][col], board[row][col+1],
                board[row][col+2], board[row][col+3]
            };
            totalScore += evaluateAdjacents(adjacentPieces, player);
        }
    }

    // Vertical evaluation
    #pragma omp parallel for reduction(+:totalScore)
    for (int col = 0; col < COLUMNS; col++) {
        for (int row = 0; row < ROWS - MAX_SPACE_TO_WIN; row++) {
            std::vector<int> adjacentPieces = {
                board[row][col], board[row+1][col],
                board[row+2][col], board[row+3][col]
            };
            totalScore += evaluateAdjacents(adjacentPieces, player);
        }
    }

    // Diagonal evaluations
    #pragma omp parallel for reduction(+:totalScore)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = 0; row < ROWS - MAX_SPACE_TO_WIN; row++) {
            std::vector<int> adjacentPieces = {
                board[row][col], board[row+1][col+1],
                board[row+2][col+2], board[row+3][col+3]
            };
            totalScore += evaluateAdjacents(adjacentPieces, player);
        }
    }

    #pragma omp parallel for reduction(+:totalScore)
    for (int col = 0; col < COLUMNS - MAX_SPACE_TO_WIN; col++) {
        for (int row = MAX_SPACE_TO_WIN; row < ROWS; row++) {
            std::vector<int> adjacentPieces = {
                board[row][col], board[row-1][col+1],
                board[row-2][col+2], board[row-3][col+3]
            };
            totalScore += evaluateAdjacents(adjacentPieces, player);
        }
    }

    return totalScore;
}

std::pair<int, int> minimax(Board& board, int ply, bool maxiPlayer) {
    std::vector<int> validCols = validLocations(board);
    bool isTerminal = isTerminalBoard(board);

    if (ply == 0 || isTerminal) {
        if (isTerminal) {
            if (detectWin(board, HUMAN)) return {-1, -1000000000};
            else if (detectWin(board, AI)) return {-1, 1000000000};
            else return {-1, 0};
        }
        return {-1, score(board, AI)};
    }

    if (maxiPlayer) {
        int value = INT_MIN;
        int col = validCols[rand() % validCols.size()];

        #pragma omp parallel for
        for (size_t i = 0; i < validCols.size(); i++) {
            int c = validCols[i];
            Board nextBoard = cloneAndPlacePiece(board, AI, c);
            int newScore = minimax(nextBoard, ply - 1, false).second;
            
            #pragma omp critical
            {
                if (newScore > value) {
                    value = newScore;
                    col = c;
                }
            }
        }
        return {col, value};
    } else {
        int value = INT_MAX;
        int col = validCols[rand() % validCols.size()];

        #pragma omp parallel for
        for (size_t i = 0; i < validCols.size(); i++) {
            int c = validCols[i];
            Board nextBoard = cloneAndPlacePiece(board, HUMAN, c);
            int newScore = minimax(nextBoard, ply - 1, true).second;
            
            #pragma omp critical
            {
                if (newScore < value) {
                    value = newScore;
                    col = c;
                }
            }
        }
        return {col, value};
    }
}

void drawGame(const Board& board, int turn, bool gameOver = false, int aiMove = 0, double runningTime = 0) {
    system("cls");
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    
    // Title with color
    SetConsoleTextAttribute(hConsole, 11); // Light cyan
    std::cout << "\n";
    std::cout << "  _____                            _     _____                \n";
    std::cout << " / ____|                          | |   |  ___|__  _   _ _ __ \n";
    std::cout << "| |     ___  _ __  _ __   ___  ___| |_  | |_ / _ \\| | | | '__|\n";
    std::cout << "| |    / _ \\| '_ \\| '_ \\ / _ \\/ __| __| |  _| (_) | |_| | |   \n";
    std::cout << "| |___| (_) | | | | | | |  __/ (__| |_  |_|  \\___/ \\__,_|_|   \n";
    std::cout << " \\_____\\___/|_| |_|_| |_|\\___|\\___|\\__|                     \n\n";

    // Game status box
    SetConsoleTextAttribute(hConsole, 14); // Yellow
    std::cout << "                     +---------------------------+\n";
    std::cout << "                     |                           |\n";
    
    // Status message
    SetConsoleTextAttribute(hConsole, 15); // White
    if (turn == HUMAN && !gameOver)
        std::cout << "                     |        Your turn!        |\n";
    else if (turn == AI && !gameOver)
        std::cout << "                     |     Computer's turn      |\n";
    else if (turn == HUMAN && gameOver)
        std::cout << "                     |       You win!! :)       |\n";
    else if (turn == AI && gameOver)
        std::cout << "                     |     Computer wins :(     |\n";
    
    SetConsoleTextAttribute(hConsole, 14); // Yellow
    std::cout << "                     |                           |\n";
    
    // Game board
    for (const auto& row : board) {
        std::cout << "                     |  ";
        for (size_t col = 0; col < row.size(); col++) {
            if (row[col] == HUMAN) {
                SetConsoleTextAttribute(hConsole, 12); // Red
                std::cout << "O ";
            }
            else if (row[col] == AI) {
                SetConsoleTextAttribute(hConsole, 9); // Blue
                std::cout << "X ";
            }
            else {
                SetConsoleTextAttribute(hConsole, 8); // Gray
                std::cout << ". ";
            }
        }
        SetConsoleTextAttribute(hConsole, 14); // Yellow
        std::cout << " |\n";
    }
    
    // Column numbers
    std::cout << "                     |  1 2 3 4 5 6 7          |\n";
    std::cout << "                     +---------------------------+\n\n";
    
    if (!gameOver) {
        SetConsoleTextAttribute(hConsole, 15); // White
        std::cout << "              Type column number (1-7) or 'q' to quit\n";
        if (turn == HUMAN) {
            SetConsoleTextAttribute(hConsole, 10); // Green
            std::cout << "             Minimax running time: " << std::fixed << std::setprecision(4) 
                      << runningTime << " seconds\n";
            SetConsoleTextAttribute(hConsole, 15); // White
            std::cout << "Your move: ";
        } else {
            SetConsoleTextAttribute(hConsole, 13); // Magenta
            std::cout << "\nComputer is thinking...\n";
        }
    }
    
    // Reset color
    SetConsoleTextAttribute(hConsole, 7); // Default gray
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    Board board = createBoard();
    int turn = (rand() % 2) ? HUMAN : AI;
    bool isGameWon = false;
    int aiMove = -1;
    double runningTime = 0;
    int totalMoves = 0;
    std::vector<double> minimaxTimes;

    drawGame(board, turn);

    while (!isGameWon) {
        totalMoves++;
        if (turn == HUMAN) {
            std::string input;
            std::cin >> input;
            
            if (input == "q") {
                system("cls");
                std::cout << "\nThank you for playing!\n";
                return 0;
            }

            try {
                int column = std::atoi(input.c_str());
                if (column >= 1 && column <= 7 && isValidColumn(board, column)) {
                    placePiece(board, HUMAN, column);
                    isGameWon = detectWin(board, turn);
                    if (isGameWon) {
                        drawGame(board, turn, true, aiMove, runningTime);
                        break;
                    }
                    turn = AI;
                    drawGame(board, turn, false, aiMove, runningTime);
                    continue;
                }
            } catch (...) {}

            std::cout << "\nInvalid input, try again...\n";
            Sleep(1000);
            drawGame(board, turn, false, aiMove, runningTime);
        } else {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            auto [move, value] = minimax(board, DEPTH, true);
            aiMove = move;
            placePiece(board, AI, aiMove);
            isGameWon = detectWin(board, AI);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            runningTime = std::chrono::duration<double>(endTime - startTime).count();
            minimaxTimes.push_back(runningTime);

            if (isGameWon) {
                drawGame(board, turn, true, aiMove, runningTime);
                break;
            }
            turn = HUMAN;
            drawGame(board, turn, false, aiMove, runningTime);
        }
    }

    if (isGameWon) {
        double avgTime = 0;
        for (double t : minimaxTimes) avgTime += t;
        avgTime /= minimaxTimes.size();
        
        std::cout << "                    Thank you for playing!\n";
        std::cout << "          Average minimax running time: " << std::fixed 
                  << std::setprecision(4) << avgTime << " seconds\n";
        std::cout << "                   Total number of moves: " << totalMoves << "\n";
    }

    return 0;
} 
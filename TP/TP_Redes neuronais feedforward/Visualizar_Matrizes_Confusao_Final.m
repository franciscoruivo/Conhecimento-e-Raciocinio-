% Script para visualizar matrizes de confusão - Comparação Final
clear all;
close all;
clc;

% Definir as classes
classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};

% Função para ler e processar matriz de confusão
function matriz = lerMatrizConfusao(filename)
    % Ler todas as linhas do ficheiro
    fid = fopen(filename, 'r');
    conteudo = fileread(filename);
    fclose(fid);
    
    % Dividir em linhas
    linhas = strsplit(conteudo, '\n');
    
    % Inicializar matriz
    matriz = zeros(6,6);
    
    % Processar cada linha com dados (começando após o cabeçalho)
    for i = 1:6
        % Pegar a linha com os números (pulando as linhas de cabeçalho)
        linha_atual = linhas{i + 3};  % Ajustado para pular cabeçalho
        
        % Remover o nome da classe do início
        partes = strsplit(linha_atual, ' ');
        numeros = [];
        
        % Extrair os números, ignorando espaços vazios
        for j = 1:length(partes)
            num_str = partes{j};
            if ~isempty(num_str) && contains(num_str, '%')
                % Remover o % e converter para número
                num = str2double(num_str(1:end-1));
                if ~isnan(num)
                    numeros = [numeros, num];
                end
            end
        end
        
        % Garantir que temos 6 números
        if length(numeros) == 6
            matriz(i,:) = numeros;
        end
    end
end

% Carregar todas as matrizes dos ficheiros
matriz_conf_rede_1_start = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_1_start.txt');
matriz_conf_rede_1_train = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_1_train.txt');
matriz_conf_rede_1_test = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_1_test.txt');
matriz_conf_rede_2_start = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_2_start.txt');
matriz_conf_rede_2_train = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_2_train.txt');
matriz_conf_rede_2_test = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_2_test.txt');
matriz_conf_rede_3_start = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_3_start.txt');
matriz_conf_rede_3_train = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_3_train.txt');
matriz_conf_rede_3_test = lerMatrizConfusao('resultados_alinea_c_ii/matriz_conf_rede_3_test.txt');

% Figura 1: Comparação dos conjuntos de teste das 3 redes
figure('Position', [100 100 1500 500]);
subplot(1,3,1);
confusionchart(matriz_conf_rede_1_test, classes, 'Title', 'Rede 1 (80/10/10) - Teste', 'RowSummary', 'row-normalized');
subplot(1,3,2);
confusionchart(matriz_conf_rede_2_test, classes, 'Title', 'Rede 2 (trainlm) - Teste', 'RowSummary', 'row-normalized');
subplot(1,3,3);
confusionchart(matriz_conf_rede_3_test, classes, 'Title', 'Rede 3 ([10 10]) - Teste', 'RowSummary', 'row-normalized');
sgtitle('Comparação das Matrizes de Confusão - Conjuntos de Teste');
print('Comparacao_Testes_Redes', '-dpng', '-r300');

% Figura 2: Evolução Rede 1
figure('Position', [100 100 1500 500]);
subplot(1,3,1);
confusionchart(matriz_conf_rede_1_start, classes, 'Title', 'Start', 'RowSummary', 'row-normalized');
subplot(1,3,2);
confusionchart(matriz_conf_rede_1_train, classes, 'Title', 'Train', 'RowSummary', 'row-normalized');
subplot(1,3,3);
confusionchart(matriz_conf_rede_1_test, classes, 'Title', 'Test', 'RowSummary', 'row-normalized');
sgtitle('Evolução da Rede 1 (80/10/10)');
print('Evolucao_Rede_1', '-dpng', '-r300');

% Figura 3: Evolução Rede 2
figure('Position', [100 100 1500 500]);
subplot(1,3,1);
confusionchart(matriz_conf_rede_2_start, classes, 'Title', 'Start', 'RowSummary', 'row-normalized');
subplot(1,3,2);
confusionchart(matriz_conf_rede_2_train, classes, 'Title', 'Train', 'RowSummary', 'row-normalized');
subplot(1,3,3);
confusionchart(matriz_conf_rede_2_test, classes, 'Title', 'Test', 'RowSummary', 'row-normalized');
sgtitle('Evolução da Rede 2 (trainlm)');
print('Evolucao_Rede_2', '-dpng', '-r300');

% Figura 4: Evolução Rede 3
figure('Position', [100 100 1500 500]);
subplot(1,3,1);
confusionchart(matriz_conf_rede_3_start, classes, 'Title', 'Start', 'RowSummary', 'row-normalized');
subplot(1,3,2);
confusionchart(matriz_conf_rede_3_train, classes, 'Title', 'Train', 'RowSummary', 'row-normalized');
subplot(1,3,3);
confusionchart(matriz_conf_rede_3_test, classes, 'Title', 'Test', 'RowSummary', 'row-normalized');
sgtitle('Evolução da Rede 3 ([10 10])');
print('Evolucao_Rede_3', '-dpng', '-r300'); 
tempoInicioScript = tic;

% Carregar imagens da pasta "test"
classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
imgSize = [28 28];
pathTest = 'test/';

fprintf('A carregar imagens da pasta "test"...\n');
Xtest = [];
Ytest = [];
for i = 1:length(classes)
    pasta_classe = fullfile(pathTest, classes{i});
    ficheiros = dir(fullfile(pasta_classe, '*.png'));
    fprintf('A carregar %d imagens da classe %s\n', length(ficheiros), classes{i});
    for j = 1:length(ficheiros)
        nome_ficheiro = fullfile(pasta_classe, ficheiros(j).name);
        img = imread(nome_ficheiro);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img_bin = imbinarize(img);
        img_redim = imresize(img_bin, imgSize);
        Xtest = [Xtest, double(img_redim(:))];
        y = zeros(length(classes), 1);
        y(i) = 1;
        Ytest = [Ytest, y];
    end
end

% Criar pasta para resultados
pastResultados = 'resultados_alinea_c_i';
if ~exist(pastResultados, 'dir')
    mkdir(pastResultados);
end

% Procurar as redes guardadas
fprintf('A procurar as redes treinadas...\n');
caminho_resultados = fullfile(pwd, 'Melhores Redes');

if ~exist(caminho_resultados, 'dir')
    caminho_resultados = fullfile(pwd, 'resultados_alinea_b');
end

if ~exist(caminho_resultados, 'dir')
    mkdir(caminho_resultados);
    warning('Pasta %s não encontrada. Foi criada, mas é necessário copiar as redes para esta localização.', caminho_resultados);
end

fprintf('A usar pasta: %s\n', caminho_resultados);

caminhos_redes = {
    fullfile(caminho_resultados, 'melhor_rede_1.mat'),
    fullfile(caminho_resultados, 'melhor_rede_2.mat'),
    fullfile(caminho_resultados, 'melhor_rede_3.mat')
};

% Testar cada rede
resultados = struct('rede', {}, 'precisao', {});
for i = 1:length(caminhos_redes)
    try
        fprintf('\nA testar rede %d: %s\n', i, caminhos_redes{i});
        dados = load(caminhos_redes{i});
        net = dados.net;
        
        % Combinar todas as imagens para treino
        imagens_todas = [imagens_start, imagens_train, imagens_test];
        targets_todas = [targets_start, targets_train, targets_test];
        
        % Treinar rede usando TODAS as imagens
        [net, tr] = train(net, imagens_todas, targets_todas);
        
        saidas = net(Xtest);
        [~, predicoes] = max(saidas);
        [~, reais] = max(Ytest);
        
        precisao = sum(predicoes == reais) / length(reais) * 100;
        fprintf('Precisão global: %.2f%%\n', precisao);
        
        fprintf('Precisão por classe:\n');
        for c = 1:length(classes)
            idx_classe = find(reais == c);
            acertos = sum(predicoes(idx_classe) == c);
            precisao_classe = acertos / length(idx_classe) * 100;
            fprintf('  - %s: %.2f%% (%d/%d)\n', classes{c}, precisao_classe, acertos, length(idx_classe));
        end
        
        % Matriz de confusão
        matrizConf = zeros(length(classes));
        for j = 1:length(predicoes)
            matrizConf(reais(j), predicoes(j)) = matrizConf(reais(j), predicoes(j)) + 1;
        end
        
        % Mostrar matriz de confusão na consola
        fprintf('Matriz de confusão (contagens):\n');
        for r = 1:size(matrizConf, 1)
            for c = 1:size(matrizConf, 2)
                fprintf('%4d ', matrizConf(r, c));
            end
            fprintf('| %s\n', classes{r});
        end
        fprintf('\n');
        
        % Calcular matriz em percentagens
        matrizConfPerc = zeros(size(matrizConf));
        for r = 1:size(matrizConf, 1)
            if sum(matrizConf(r,:)) > 0
                matrizConfPerc(r,:) = matrizConf(r,:) / sum(matrizConf(r,:)) * 100;
            end
        end
        
        % Guardar matriz em formato de texto
        try
            % Nome do ficheiro
            ficheiro_matriz = fullfile(pastResultados, ['matriz_conf_rede_' num2str(i) '.txt']);
            fileID = fopen(ficheiro_matriz, 'w');
            
            % Escrever cabeçalho
            fprintf(fileID, 'MATRIZ DE CONFUSÃO - REDE %d\n\n', i);
            fprintf(fileID, '%-12s ', 'Classe Real');
            for c = 1:length(classes)
                fprintf(fileID, '%-12s ', classes{c});
            end
            fprintf(fileID, '\n');
            fprintf(fileID, '%-12s ', '-----------');
            for c = 1:length(classes)
                fprintf(fileID, '%-12s ', '------------');
            end
            fprintf(fileID, '\n');
            
            % Escrever linhas da matriz
            for r = 1:size(matrizConfPerc, 1)
                fprintf(fileID, '%-12s ', classes{r});
                for c = 1:size(matrizConfPerc, 2)
                    fprintf(fileID, '%10.1f%% ', matrizConfPerc(r,c));
                end
                fprintf(fileID, '\n');
            end
            
            fclose(fileID);
            fprintf('Matriz de confusão guardada em: %s\n', ficheiro_matriz);
            
            % Nota: Visualização gráfica temporariamente desativada devido a problemas técnicos
            % As matrizes de confusão estão disponíveis em formato texto nos ficheiros .txt
            
        catch ME
            fprintf('Erro ao guardar matriz: %s\n', ME.message);
        end
        
        resultados(i).rede = ['Rede ' num2str(i)];
        resultados(i).precisao = precisao;
        
    catch ME
        fprintf('Erro ao testar a rede %d: %s\n', i, ME.message);
    end
end

% Guardar resultados em Excel
try
    T = struct2table(resultados);
    writetable(T, fullfile(pastResultados, 'resultados_alinea_c_i.xlsx'));
catch ME
    fprintf('Erro ao guardar Excel: %s\n', ME.message);
    for i = 1:length(resultados)
        fprintf('Rede %d: Precisão = %.2f%%\n', i, resultados(i).precisao);
    end
end

% Comparação com tarefa B
fprintf('\n== COMPARAÇÃO COM RESULTADOS DA TAREFA B ==\n');
fprintf('Na tarefa B, as melhores configurações obtiveram:\n');
fprintf('- Rede 1 (divisão 80/10/10): Precisão Teste = 71,00%%\n');
fprintf('- Rede 2 (algoritmo trainlm): Precisão Teste = 70,67%%\n');
fprintf('- Rede 3 (topologia [10 10]): Precisão Teste = 67,11%%\n\n');

fprintf('Na tarefa C-i (teste com novas imagens), obtemos:\n');
for i = 1:length(resultados)
    fprintf('- %s: Precisão = %.2f%%\n', resultados(i).rede, resultados(i).precisao);
end

% Análise comparativa
precisoes_b = [71.00, 70.67, 67.11];
precisoes_c = zeros(1, length(resultados));

for i = 1:length(resultados)
    if i <= length(precisoes_b)
        precisoes_c(i) = resultados(i).precisao;
    end
end

% Conclusões
fprintf('\n== CONCLUSÕES ==\n');
if isempty(resultados) || length(resultados) < 3
    fprintf('Não foi possível obter resultados suficientes.\n');
    fprintf('Verifique se as redes estão disponíveis no caminho correto.\n');
else
    media_b = mean(precisoes_b);
    media_c = mean(precisoes_c);
    diferenca_media = media_c - media_b;
    
    fprintf('Análise do desempenho das redes:\n');
    fprintf('- Média de precisão na tarefa B: %.2f%%\n', media_b);
    fprintf('- Média de precisão na tarefa C-i: %.2f%%\n', media_c);
    fprintf('- Diferença média: %.2f%%\n', diferenca_media);
    
    if diferenca_media < -5
        fprintf('\nHouve uma redução significativa no desempenho com as novas imagens.\n');
    elseif diferenca_media < 0
        fprintf('\nHouve uma ligeira redução no desempenho com as novas imagens.\n');
    else
        fprintf('\nO desempenho manteve-se ou melhorou com as novas imagens.\n');
    end
end

tempoFinal = toc(tempoInicioScript);
fprintf('\nTempo total de execução: %.2f segundos\n', tempoFinal);

% Combinar todas as imagens para treino
imagens_todas = [imagens_start, imagens_train, imagens_test];
targets_todas = [targets_start, targets_train, targets_test];

% Guardar as três melhores redes obtidas nesta experimentação
copyfile('resultados_alinea_c_iii/rede_1_todas.mat', 'Melhores Redes Finais/rede_final_1.mat');
copyfile('resultados_alinea_c_iii/rede_2_todas.mat', 'Melhores Redes Finais/rede_final_2.mat');
copyfile('resultados_alinea_c_iii/rede_3_todas.mat', 'Melhores Redes Finais/rede_final_3.mat');

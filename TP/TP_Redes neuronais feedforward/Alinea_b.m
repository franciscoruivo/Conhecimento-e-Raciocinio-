clear all;
close all;
clc;

% Configuração inicial
current_dir = pwd;
pasta_train = fullfile(current_dir, 'train');

% Verificar se a pasta existe
if ~exist(pasta_train, 'dir')
    error('A pasta %s não existe! Verifique o caminho.', pasta_train);
end

classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};  
num_classes = length(classes);

% Criar pasta para guardar os resultados
resultsDir = 'resultados_tarefa_b';
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Carregar e processar todas as imagens da pasta train
[inputs, targets] = carregar_imagens(pasta_train, classes);

% Função para executar treino e avaliação com diferentes configurações
function [mediaAccuracy, mediaTestAccuracy, confusionMatrix] = executar_configuracao(inputs, targets, topologia, activationFunctions, trainFunc, divideRatios, numRepeticoes, resultsDir)
    mediaAccuracy = 0;
    mediaTestAccuracy = 0;
    confusionMatrix = zeros(size(targets, 1), size(targets, 1));
    
    for i = 1:numRepeticoes
        % Criar a rede com a topologia especificada
        net = feedforwardnet(topologia);
        
        % Configurar função de treino
        net.trainFcn = trainFunc;
        
        % Configurar funções de ativação para cada camada
        for j = 1:length(topologia)
            if j <= length(activationFunctions)
                net.layers{j}.transferFcn = activationFunctions{j};
            end
        end
        
        % Configurar divisão dos dados
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = divideRatios(1);
        net.divideParam.valRatio = divideRatios(2);
        net.divideParam.testRatio = divideRatios(3);
        
        % Aumentar número de épocas para treino mais completo
        net.trainParam.epochs = 500;
        net.trainParam.goal = 1e-6;
        % Ajustar taxa de aprendizagem
        net.trainParam.lr = 0.01;
        net.trainParam.showWindow = true; % Mostrar janela de progresso durante o treino
        
        % Treinar a rede
        [net, tr] = train(net, inputs, targets);
        
        % Avaliar desempenho global
        outputs = net(inputs);
        
        % Calcular precisão global
        r = 0;
        for j = 1:size(outputs, 2)
            [~, predClass] = max(outputs(:, j));
            [~, trueClass] = max(targets(:, j));
            if predClass == trueClass
                r = r + 1;
            end
            confusionMatrix(trueClass, predClass) = confusionMatrix(trueClass, predClass) + 1;
        end
        accuracyGlobal = r / size(outputs, 2) * 100;
        
        % Calcular precisão de teste
        testIndices = tr.testInd;
        if ~isempty(testIndices)
            testOutputs = net(inputs(:, testIndices));
            testTargets = targets(:, testIndices);
            
            rTest = 0;
            for j = 1:size(testOutputs, 2)
                [~, predClass] = max(testOutputs(:, j));
                [~, trueClass] = max(testTargets(:, j));
                if predClass == trueClass
                    rTest = rTest + 1;
                end
            end
            accuracyTest = rTest / size(testOutputs, 2) * 100;
        else
            accuracyTest = 0;
        end
        
        fprintf('Execução %d: Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', i, accuracyGlobal, accuracyTest);
        
        mediaAccuracy = mediaAccuracy + accuracyGlobal;
        mediaTestAccuracy = mediaTestAccuracy + accuracyTest;
        
        % Calcular matriz de confusão em percentagem
        confusionMatrixPercent = zeros(size(confusionMatrix));
        for r = 1:size(confusionMatrix, 1)
            rowSum = sum(confusionMatrix(r, :));
            if rowSum > 0
                confusionMatrixPercent(r, :) = confusionMatrix(r, :) / rowSum * 100;
            end
        end
        
        % Guardar detalhes da matriz para análise
        if i == numRepeticoes
            % Na última repetição, guarda os detalhes
            figure;
            imagesc(confusionMatrixPercent);
            colormap('jet');
            colorbar;
            title(['Matriz de Confusão - ' num2str(topologia)]);
            xlabel('Previsto');
            ylabel('Real');
            set(gca, 'XTick', 1:size(confusionMatrix, 1), 'XTickLabel', {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'}, ...
                     'YTick', 1:size(confusionMatrix, 1), 'YTickLabel', {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'});
            % Usar print em vez de saveas para evitar erro
            print(gcf, fullfile(resultsDir, ['matriz_' trainFunc '_' num2str(i)]), '-dpng');
            close(gcf); % Fechar a figura após guardar
        end
    end
    
    mediaAccuracy = mediaAccuracy / numRepeticoes;
    mediaTestAccuracy = mediaTestAccuracy / numRepeticoes;
    confusionMatrix = confusionMatrix / numRepeticoes;
end

% Parte B-i: Comparar diferentes topologias
disp('===== A testar diferentes topologias =====');

% Testar diferentes números de neurónios e camadas
topologias = {
    10,           % 1 camada com 10 neurónios
    [5, 5],         % 2 camadas com 5 neurónios cada
    [10, 10],       % 2 camadas com 10 neurónios cada
    [5, 5, 5],      % 3 camadas com 5 neurónios cada
    [10, 10, 10]    % 3 camadas com 10 neurónios cada
};

resultados_topologias = struct('topologia', {}, 'mediaAccuracy', {}, 'mediaTestAccuracy', {}, 'confusionMatrix', {});

for i = 1:length(topologias)
    fprintf('\nA testar topologia: ');
    disp(topologias{i});
    
    % Usar configurações padrão para outras propriedades
    activationFunctions = repmat({'tansig'}, 1, length(topologias{i}));
    activationFunctions{end} = 'purelin';  % Função de ativação da camada de saída
    
    [mediaAccuracy, mediaTestAccuracy, confusionMatrix] = executar_configuracao(inputs, targets, topologias{i}, activationFunctions, 'trainlm', [0.7, 0.15, 0.15], 10, resultsDir);
    
    fprintf('Topologia %d: Média Precisão Global = %.2f%%, Média Precisão Teste = %.2f%%\n', i, mediaAccuracy, mediaTestAccuracy);
    
    % Guardar resultados
    resultados_topologias(i).topologia = topologias{i};
    resultados_topologias(i).mediaAccuracy = mediaAccuracy;
    resultados_topologias(i).mediaTestAccuracy = mediaTestAccuracy;
    resultados_topologias(i).confusionMatrix = confusionMatrix;
    
    % Guardar a rede com o melhor desempenho desta topologia
    net = feedforwardnet(topologias{i});
    for j = 1:length(topologias{i})
        if j <= length(activationFunctions)
            net.layers{j}.transferFcn = activationFunctions{j};
        end
    end
    net.trainFcn = 'trainlm';
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    [net, ~] = train(net, inputs, targets);
    save(fullfile(resultsDir, sprintf('rede_topologia_%d.mat', i)), 'net');
end

% Parte B-ii: Comparar diferentes funções de ativação
disp('===== A testar diferentes funções de ativação =====');

% Usar a melhor topologia encontrada
melhor_topologia_idx = find([resultados_topologias.mediaAccuracy] == max([resultados_topologias.mediaAccuracy]), 1);
melhor_topologia = topologias{melhor_topologia_idx};

funcoes_ativacao = {
    {'logsig', 'purelin'},
    {'tansig', 'logsig'},
    {'logsig', 'radbasn'},
    {'radbas', 'softmax'}
};

resultados_ativacao = struct('funcoes', {}, 'mediaAccuracy', {}, 'mediaTestAccuracy', {}, 'confusionMatrix', {});

for i = 1:length(funcoes_ativacao)
    fprintf('\nA testar funções de ativação: ');
    disp(funcoes_ativacao{i});
    
    % Ajustar o número de funções para corresponder à topologia
    funcs = funcoes_ativacao{i};
    if length(funcs) < length(melhor_topologia)
        funcs = [repmat(funcs(1), 1, length(melhor_topologia) - 1), funcs(end)];
    elseif length(funcs) > length(melhor_topologia)
        funcs = funcs(1:length(melhor_topologia));
    end
    
    [mediaAccuracy, mediaTestAccuracy, confusionMatrix] = executar_configuracao(inputs, targets, melhor_topologia, funcs, 'trainlm', [0.7, 0.15, 0.15], 10, resultsDir);
    
    fprintf('Funções %d: Média Precisão Global = %.2f%%, Média Precisão Teste = %.2f%%\n', i, mediaAccuracy, mediaTestAccuracy);
    
    % Guardar resultados
    resultados_ativacao(i).funcoes = funcs;
    resultados_ativacao(i).mediaAccuracy = mediaAccuracy;
    resultados_ativacao(i).mediaTestAccuracy = mediaTestAccuracy;
    resultados_ativacao(i).confusionMatrix = confusionMatrix;
    
    % Guardar a rede com o melhor desempenho desta configuração
    net = feedforwardnet(melhor_topologia);
    for j = 1:length(melhor_topologia)
        if j <= length(funcs)
            net.layers{j}.transferFcn = funcs{j};
        end
    }
    net.trainFcn = 'trainlm';
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    [net, ~] = train(net, inputs, targets);
    save(fullfile(resultsDir, sprintf('rede_ativacao_%d.mat', i)), 'net');
end

% Parte B-iii: Comparar diferentes funções de treino
disp('===== A testar diferentes funções de treino =====');

% Usar a melhor topologia e as melhores funções de ativação
melhor_ativacao_idx = find([resultados_ativacao.mediaAccuracy] == max([resultados_ativacao.mediaAccuracy]), 1);
melhor_ativacao = resultados_ativacao(melhor_ativacao_idx).funcoes;

funcoes_treino = {'trainlm', 'traingd', 'trainbr', 'traincgf'};

resultados_treino = struct('funcao', {}, 'mediaAccuracy', {}, 'mediaTestAccuracy', {}, 'confusionMatrix', {});

for i = 1:length(funcoes_treino)
    fprintf('\nA testar a função de treino: %s\n', funcoes_treino{i});
    
    try
        [mediaAccuracy, mediaTestAccuracy, confusionMatrix] = executar_configuracao(inputs, targets, melhor_topologia, melhor_ativacao, funcoes_treino{i}, [0.7, 0.15, 0.15], 10, resultsDir);
        
        fprintf('Função de treino %s: Média Precisão Global = %.2f%%, Média Precisão Teste = %.2f%%\n', funcoes_treino{i}, mediaAccuracy, mediaTestAccuracy);
        
        % Guardar resultados
        resultados_treino(i).funcao = funcoes_treino{i};
        resultados_treino(i).mediaAccuracy = mediaAccuracy;
        resultados_treino(i).mediaTestAccuracy = mediaTestAccuracy;
        resultados_treino(i).confusionMatrix = confusionMatrix;
        
        % Guardar a rede com o melhor desempenho desta configuração
        net = feedforwardnet(melhor_topologia);
        for j = 1:length(melhor_topologia)
            if j <= length(melhor_ativacao)
                net.layers{j}.transferFcn = melhor_ativacao{j};
            end
        end
        net.trainFcn = funcoes_treino{i};
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
        [net, ~] = train(net, inputs, targets);
        save(fullfile(resultsDir, sprintf('rede_treino_%s.mat', funcoes_treino{i})), 'net');
    catch e
        fprintf('ERRO ao testar a função de treino %s: %s\n', funcoes_treino{i}, e.message);
        continue;
    end
end

% Parte B-iv: Comparar diferentes divisões de treino/validação/teste
disp('===== A testar diferentes divisões de treino/validação/teste =====');

% Usar a melhor topologia, funções de ativação e função de treino
melhor_treino_idx = find([resultados_treino.mediaAccuracy] == max([resultados_treino.mediaAccuracy]), 1);
melhor_treino = resultados_treino(melhor_treino_idx).funcao;

divisoes = {
    [0.7, 0.15, 0.15],  % 70% treino, 15% validação, 15% teste
    [0.8, 0.1, 0.1],    % 80% treino, 10% validação, 10% teste
    [0.6, 0.2, 0.2],    % 60% treino, 20% validação, 20% teste
    [0.5, 0.25, 0.25],  % 50% treino, 25% validação, 25% teste
    [0.9, 0.05, 0.05]   % 90% treino, 5% validação, 5% teste
};

resultados_divisao = struct('divisao', {}, 'mediaAccuracy', {}, 'mediaTestAccuracy', {}, 'confusionMatrix', {});

for i = 1:length(divisoes)
    fprintf('\nA testar divisão dos dados: %.2f/%.2f/%.2f\n', divisoes{i}(1), divisoes{i}(2), divisoes{i}(3));
    
    [mediaAccuracy, mediaTestAccuracy, confusionMatrix] = executar_configuracao(inputs, targets, melhor_topologia, melhor_ativacao, melhor_treino, divisoes{i}, 10, resultsDir);
    
    fprintf('Divisão %.2f/%.2f/%.2f: Média Precisão Global = %.2f%%, Média Precisão Teste = %.2f%%\n', divisoes{i}(1), divisoes{i}(2), divisoes{i}(3), mediaAccuracy, mediaTestAccuracy);
    
    % Guardar resultados
    resultados_divisao(i).divisao = divisoes{i};
    resultados_divisao(i).mediaAccuracy = mediaAccuracy;
    resultados_divisao(i).mediaTestAccuracy = mediaTestAccuracy;
    resultados_divisao(i).confusionMatrix = confusionMatrix;
    
    % Guardar a rede com o melhor desempenho desta configuração
    net = feedforwardnet(melhor_topologia);
    for j = 1:length(melhor_topologia)
        if j <= length(melhor_ativacao)
            net.layers{j}.transferFcn = melhor_ativacao{j};
        end
    end
    net.trainFcn = melhor_treino;
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = divisoes{i}(1);
    net.divideParam.valRatio = divisoes{i}(2);
    net.divideParam.testRatio = divisoes{i}(3);
    
    [net, ~] = train(net, inputs, targets);
    save(fullfile(resultsDir, sprintf('rede_divisao_%d.mat', i)), 'net');
end

% Identificar e guardar as três melhores redes
disp('===== A identificar as três melhores redes =====');

% Combinar todos os resultados
todos_resultados = [];
contador = 1;

for i = 1:length(resultados_topologias)
    todos_resultados(contador).tipo = 'topologia';
    todos_resultados(contador).indice = i;
    todos_resultados(contador).mediaAccuracy = resultados_topologias(i).mediaAccuracy;
    todos_resultados(contador).mediaTestAccuracy = resultados_topologias(i).mediaTestAccuracy;
    contador = contador + 1;
end

for i = 1:length(resultados_ativacao)
    todos_resultados(contador).tipo = 'ativacao';
    todos_resultados(contador).indice = i;
    todos_resultados(contador).mediaAccuracy = resultados_ativacao(i).mediaAccuracy;
    todos_resultados(contador).mediaTestAccuracy = resultados_ativacao(i).mediaTestAccuracy;
    contador = contador + 1;
end

for i = 1:length(resultados_treino)
    todos_resultados(contador).tipo = 'treino';
    todos_resultados(contador).indice = i;
    todos_resultados(contador).mediaAccuracy = resultados_treino(i).mediaAccuracy;
    todos_resultados(contador).mediaTestAccuracy = resultados_treino(i).mediaTestAccuracy;
    contador = contador + 1;
end

for i = 1:length(resultados_divisao)
    todos_resultados(contador).tipo = 'divisao';
    todos_resultados(contador).indice = i;
    todos_resultados(contador).mediaAccuracy = resultados_divisao(i).mediaAccuracy;
    todos_resultados(contador).mediaTestAccuracy = resultados_divisao(i).mediaTestAccuracy;
    contador = contador + 1;
end

% Ordenar por precisão de teste
[~, idx] = sort([todos_resultados.mediaTestAccuracy], 'descend');
melhores_redes = todos_resultados(idx(1:3));

for i = 1:length(melhores_redes)
    tipo = melhores_redes(i).tipo;
    indice = melhores_redes(i).indice;
    
    fprintf('Melhor rede %d: Tipo = %s, Índice = %d, Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', ...
        i, tipo, indice, melhores_redes(i).mediaAccuracy, melhores_redes(i).mediaTestAccuracy);
    
    % Copiar a rede correspondente para a pasta das melhores redes
    switch tipo
        case 'topologia'
            copyfile(fullfile(resultsDir, sprintf('rede_topologia_%d.mat', indice)), ...
                    fullfile(resultsDir, sprintf('melhor_rede_%d.mat', i)));
        case 'ativacao'
            copyfile(fullfile(resultsDir, sprintf('rede_ativacao_%d.mat', indice)), ...
                    fullfile(resultsDir, sprintf('melhor_rede_%d.mat', i)));
        case 'treino'
            copyfile(fullfile(resultsDir, sprintf('rede_treino_%s.mat', resultados_treino(indice).funcao)), ...
                    fullfile(resultsDir, sprintf('melhor_rede_%d.mat', i)));
        case 'divisao'
            copyfile(fullfile(resultsDir, sprintf('rede_divisao_%d.mat', indice)), ...
                    fullfile(resultsDir, sprintf('melhor_rede_%d.mat', i)));
    end
end
disp('===== Resultados Finais =====');
disp('Melhores topologias:');
for i = 1:length(resultados_topologias)
    fprintf('Topologia %d: %s, Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', ...
        i, mat2str(resultados_topologias(i).topologia), resultados_topologias(i).mediaAccuracy, resultados_topologias(i).mediaTestAccuracy);
end

disp('Melhores funções de ativação:');
for i = 1:length(resultados_ativacao)
    fprintf('Função %d: %s, Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', ...
        i, strjoin(resultados_ativacao(i).funcoes, '/'), resultados_ativacao(i).mediaAccuracy, resultados_ativacao(i).mediaTestAccuracy);
end

disp('Melhores funções de treino:');
for i = 1:length(resultados_treino)
    fprintf('Função %s: Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', ...
        resultados_treino(i).funcao, resultados_treino(i).mediaAccuracy, resultados_treino(i).mediaTestAccuracy);
end

disp('Melhores divisões:');
for i = 1:length(resultados_divisao)
    fprintf('Divisão %.2f/%.2f/%.2f: Precisão Global = %.2f%%, Precisão Teste = %.2f%%\n', ...
        resultados_divisao(i).divisao(1), resultados_divisao(i).divisao(2), resultados_divisao(i).divisao(3), ...
        resultados_divisao(i).mediaAccuracy, resultados_divisao(i).mediaTestAccuracy);
end

% Exportar resultados para Excel
T = struct2table(resultados_topologias);
writetable(T, fullfile(resultsDir, 'resultados_topologias.xlsx'));

T2 = struct2table(resultados_ativacao);
writetable(T2, fullfile(resultsDir, 'resultados_ativacao.xlsx'));

T3 = struct2table(resultados_treino);
writetable(T3, fullfile(resultsDir, 'resultados_treino.xlsx'));

T4 = struct2table(resultados_divisao);
writetable(T4, fullfile(resultsDir, 'resultados_divisao.xlsx'));

% Verificar as redes finais guardadas
disp('===== A verificar desempenho das melhores redes =====');
for i = 1:3
    fprintf('\nA testar melhor_rede_%d.mat com os dados de treino\n', i);
    
    % Carregar a rede
    try
        rede = load(fullfile(resultsDir, sprintf('melhor_rede_%d.mat', i)));
        net = rede.net;
        
        % Testar com todos os dados
        outputs = net(inputs);
        [~, predicoes] = max(outputs);
        [~, reais] = max(targets);
        
        % Calcular precisão global
        acertos = sum(predicoes == reais);
        precisao = acertos / length(reais) * 100;
        
        fprintf('Precisão global: %.2f%%\n', precisao);
        
        % Calcular precisão por classe
        for c = 1:length(classes)
            idx = find(reais == c);
            acertos_classe = sum(predicoes(idx) == c);
            precisao_classe = acertos_classe / length(idx) * 100;
            fprintf('  - Classe %s: %.2f%% (%d/%d)\n', classes{c}, precisao_classe, acertos_classe, length(idx));
        end
        
        % Calcular matriz de confusão
        matrizConfusao = zeros(length(classes));
        for j = 1:length(predicoes)
            matrizConfusao(reais(j), predicoes(j)) = matrizConfusao(reais(j), predicoes(j)) + 1;
        end
        
        % Guardar matriz de confusão usando print em vez de saveas
        figure;
        imagesc(matrizConfusao);
        colormap('jet');
        colorbar;
        title(['Matriz Final - Rede ' num2str(i)]);
        xlabel('Previsto');
        ylabel('Real');
        set(gca, 'XTick', 1:length(classes), 'XTickLabel', classes, 'YTick', 1:length(classes), 'YTickLabel', classes);
        print(gcf, fullfile(resultsDir, ['matriz_final_rede_' num2str(i)]), '-dpng');
        close(gcf); % Fechar a figura após guardar
        
    catch e
        fprintf('Erro ao testar a rede %d: %s\n', i, e.message);
    end
end

% Função para carregar e processar imagens
function [X, Y] = carregar_imagens(pasta, classes)
    X = [];
    Y = [];
    
    if ~exist(pasta, 'dir')
        error('A pasta %s não existe!', pasta);
    end
    
    for i = 1:length(classes)
        pasta_classe = fullfile(pasta, classes{i});
        
        if ~exist(pasta_classe, 'dir')
            fprintf('AVISO: A pasta %s não existe!\n', pasta_classe);
            continue;
        end
       
        arquivos = dir(fullfile(pasta_classe, '*.png'));
        fprintf('A carregar %d imagens da classe %s\n', length(arquivos), classes{i});
        
        for j = 1:length(arquivos)
            % Carregar imagem
            nome_arquivo = fullfile(pasta_classe, arquivos(j).name);
            
            try
                img = imread(nome_arquivo);
                
                % Converter para binário e redimensionar
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                img_bin = imbinarize(img);
                img_resized = imresize(img_bin, [28 28]);
                
                % Converter para vetor
                X = [X, double(img_resized(:))];
                
                % Criar vetor de saída one-hot
                y = zeros(length(classes), 1);
                y(i) = 1;
                Y = [Y, y];
            catch e
                fprintf('ERRO ao processar ficheiro %s: %s\n', nome_arquivo, e.message);
                continue;
            end
        end
    end
    
    if isempty(X)
        error('Nenhuma imagem foi carregada. Verifique se o caminho está correto e se existem imagens nas pastas.');
    end
    
    fprintf('Total de %d imagens carregadas.\n', size(X, 2));
end 
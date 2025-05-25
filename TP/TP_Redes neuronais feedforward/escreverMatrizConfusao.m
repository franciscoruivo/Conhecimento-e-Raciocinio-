function escreverMatrizConfusao(filename, matriz_conf, classes, nome_rede, conjunto)
% ESCREVERMATRIZCONFUSAO - Escreve a matriz de confusão em formato texto
% 
% Parâmetros:
%   filename - Nome do arquivo onde a matriz será escrita
%   matriz_conf - Matriz de confusão em percentagem
%   classes - Células com os nomes das classes
%   nome_rede - Nome da rede
%   conjunto - Nome do conjunto (start, train ou test)

    % Abrir arquivo para escrita
    fid = fopen(filename, 'w');
    
    if fid == -1
        error('Não foi possível abrir o arquivo para escrita: %s', filename);
    end
    
    % Escrever cabeçalho
    fprintf(fid, 'MATRIZ DE CONFUSÃO - %s - Conjunto %s\n\n', nome_rede, conjunto);
    
    % Escrever cabeçalho das colunas
    fprintf(fid, 'Classe Real  ');
    for i = 1:length(classes)
        fprintf(fid, '%-12s ', classes{i});
    end
    fprintf(fid, '\n');
    
    % Linha separadora
    fprintf(fid, '-----------  ');
    for i = 1:length(classes)
        fprintf(fid, '------------ ');
    end
    fprintf(fid, '\n');
    
    % Escrever matriz de confusão
    for i = 1:length(classes)
        fprintf(fid, '%-12s ', classes{i});
        for j = 1:length(classes)
            fprintf(fid, '%8.1f%% ', matriz_conf(i, j));
        end
        fprintf(fid, '\n');
    end
    
    % Fechar arquivo
    fclose(fid);
end 
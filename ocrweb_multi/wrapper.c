/*
针对Pyinstaller目录下文件过多的问题, 使用外部exe+system调用的方式实现资源文件/依赖库分离
*/
#include <windows.h>
#include <stdio.h>

void combine(char *destination, const char *path1, const char *path2)
{
    if (path1 == NULL && path2 == NULL)
    {
        strcpy(destination, "");
    }
    else if (path2 == NULL || strlen(path2) == 0)
    {
        strcpy(destination, path1);
    }
    else if (path1 == NULL || strlen(path1) == 0)
    {
        strcpy(destination, path2);
    }
    else
    {
        strcpy(destination, path1);

        size_t idx = 0, sepIdx = 0;
        size_t size1 = strlen(path1);
        while (idx < size1)
        {
            idx++;
            if (destination[idx] == '\\' || destination[idx] == '/')
            {
                sepIdx = idx;
            }
        }
        // Trim destination: delete from last separator to end.
        destination[sepIdx + 1] = '\0';
        strcat(destination, path2);
    }
}

void main()
{
    // Set title
    system("title Rapid OCR Server");
    // Get wrapper exe path
    TCHAR path[MAX_PATH];
    GetModuleFileName(NULL, path, MAX_PATH);

    TCHAR exe_path[MAX_PATH];
    // Get real exe path from wrapper exe path
    combine(exe_path, path, "ocrweb\\main.exe");
    printf("Run real exe: %s\n", exe_path);
    // Run real exe
    system(exe_path);
}

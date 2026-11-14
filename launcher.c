// Tiny native starter for Live Translate.app. It starts translate.py.
// A real Apple silicon program here lets macOS show the microphone request for this app.
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    char raw[PATH_MAX], dir[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) != 0 || !realpath(raw, dir)) return 1;
    for (int i = 0; i < 4; i++) {            // .../Live Translate.app/Contents/MacOS/launcher -> project folder
        char *slash = strrchr(dir, '/');
        if (!slash) return 1;
        *slash = 0;
    }
    if (chdir(dir) != 0) return 1;
    // Run Python through a link inside this app. macOS then shows the overlay
    // as "Live Translate" with this app's icon, not as "Python".
    setenv("PYTHONPATH", "venv/lib/python3.12/site-packages", 1);
    int fd = open("last-run.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); }
    execl("Live Translate.app/Contents/MacOS/python", "python", "translate.py", (char *)NULL);
    return 127;
}

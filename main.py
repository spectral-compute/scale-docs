import subprocess

SCALE_STABLE_VERSION = "free-1.2.0"
SCALE_UNSTABLE_VERSION = "free-unstable-2025.02.19"

# Find the current git branch to choose stable or unstable variation.
g = subprocess.run(["git", "branch", "--show-current"], capture_output=True)
current_branch = g.stdout.decode("utf-8")[:-1]

print("Current branch: " + current_branch)

def define_env(env):
    env.variables["branch"] = current_branch
    env.variables["scale_version"] = SCALE_UNSTABLE_VERSION if current_branch == "unstable" else SCALE_STABLE_VERSION
    env.variables["repo_subdomain"] = "unstable-pkgs" if current_branch == "unstable" else "pkgs"

    @env.macro
    def checksum(url):
        g = subprocess.run(["curl", url + ".sha512"], capture_output=True)
        return g.stdout.decode("utf-8").split(" ")[0]
